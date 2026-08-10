/*
 * grape_gpu_kernels.cuh
 *
 * Families of CUDA kernels:
 * 
 * elementwise bookkeeping (fill copy, scale, axpy, and their fused copy-and-reduce variants)
 * 
 * sparse matrix-vector actions in plain CSR
 *
 * half-warp CSR
 *
 * SELL-32 storage (inner products with block reductions, and gradient post-processing)
 *
 * Kernel names carry a _kernel suffix
 * 
 * Kernels are called from grape_gpu_actions.cuh
 * 
 */

#ifndef GRAPE_GPU_KERNELS_CUH
#define GRAPE_GPU_KERNELS_CUH

#include "grape_gpu_types.cuh"

namespace {

// Fills complex vector with values
__global__ void fill_kernel(cdouble* x, int n, cdouble value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = value;
}

// Copies a complex vector
__global__ void copy_kernel(const cdouble* x, cdouble* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}

// Copies a complex vector and reduces its largest absolute value per block
__global__ void copy_max_abs_reduce_kernel(const cdouble* src, cdouble* dst,
                                           int n, double* blockMax) {
    __shared__ double sdata[THREADS];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double m = 0.0;
    while (i < n) {
        cdouble value = src[i];
        dst[i] = value;
        m = fmax(m, hypot(value.x, value.y));
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = m;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmax(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) blockMax[blockIdx.x] = sdata[0];
}

// Copies two complex vectors and reduces the largest absolute value of both
__global__ void copy_pair_max_abs_reduce_kernel(const cdouble* srcA, cdouble* dstA,
                                                int nA, const cdouble* srcB,
                                                cdouble* dstB, int nB,
                                                double* blockMax) {
    __shared__ double sdata[THREADS];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = max(nA, nB);
    double m = 0.0;
    while (i < n) {
        if (i < nA) {
            cdouble value = srcA[i];
            dstA[i] = value;
            m = fmax(m, hypot(value.x, value.y));
        }
        if (i < nB) {
            cdouble value = srcB[i];
            dstB[i] = value;
            m = fmax(m, hypot(value.x, value.y));
        }
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = m;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmax(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) blockMax[blockIdx.x] = sdata[0];
}

// Scales a complex vector by a real number
__global__ void scale_kernel(cdouble* x, int n, double s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = scale_cd(x[i], s);
}

// Adds one complex vector into another
__global__ void axpy_kernel(cdouble* y, const cdouble* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = add_cd(y[i], x[i]);
}

// Multiplies a complex vector by a complex scalar
__global__ void linear_scale_kernel(cdouble* y, const cdouble* x, int n, cdouble alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = mul_cd(alpha, x[i]);
}

// Multiplies a complex vector by a complex scalar and accumulates the result
__global__ void linear_scale_axpy_kernel(cdouble* y, const cdouble* x,
                                         cdouble* accum, int n,
                                         cdouble alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    cdouble value = mul_cd(alpha, x[i]);
    y[i] = value;
    accum[i] = add_cd(accum[i], value);
}

// Averages two complex vectors, giving the midpoint state
__global__ void midpoint_state_kernel(const cdouble* a, const cdouble* b, cdouble* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = make_cd(0.5*(a[i].x + b[i].x), 0.5*(a[i].y + b[i].y));
}

// Accumulates a scaled CSR matrix-vector product into the output vector
__global__ void spmv_accum_kernel(int nrows, const int* rowPtr, const int* colIdx,
                                  const cdouble* values, double alpha,
                                  const cdouble* x, cdouble* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;
    cdouble sum = make_cd(0.0, 0.0);
    for (int p = rowPtr[row]; p < rowPtr[row+1]; ++p) {
        sum = add_cd(sum, mul_cd(values[p], x[colIdx[p]]));
    }
    y[row] = add_cd(y[row], scale_cd(sum, alpha));
}

// Accumulates a scaled CSR product for a batch of vectors sharing one matrix
__global__ void spmv_accum_batch_same_matrix_kernel(int nrows, int nvec,
                                                    const int* rowPtr, const int* colIdx,
                                                    const cdouble* values, double alpha,
                                                    const cdouble* x, cdouble* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nrows * nvec;
    if (idx >= total) return;
    int row = idx % nrows;
    int vec = idx / nrows;
    const cdouble* xVec = x + static_cast<size_t>(vec) * nrows;
    cdouble* yVec = y + static_cast<size_t>(vec) * nrows;
    cdouble sum = make_cd(0.0, 0.0);
    for (int p = rowPtr[row]; p < rowPtr[row+1]; ++p) {
        sum = add_cd(sum, mul_cd(values[p], xVec[colIdx[p]]));
    }
    yVec[row] = add_cd(yVec[row], scale_cd(sum, alpha));
}

// Applies an affine Liouvillian assembled on the fly from the fused CSR value blocks
__global__ void fused_affine_spmv_kernel(int nrows, int nnz, int nctrls, int driftIndex,
                                         const int* rowPtr, const int* colIdx,
                                         const cdouble* driftValues,
                                         const cdouble* controlValues,
                                         const double* coeff,
                                         const cdouble* x, cdouble* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;
    const size_t driftBase = static_cast<size_t>(driftIndex) * nnz;
    cdouble sum = make_cd(0.0, 0.0);
    for (int p = rowPtr[row]; p < rowPtr[row+1]; ++p) {
        cdouble value = driftValues[driftBase + p];
        for (int k = 0; k < nctrls; ++k) {
            double c = coeff[k];
            if (c != 0.0) {
                value = add_cd(value, scale_cd(controlValues[static_cast<size_t>(k) * nnz + p], c));
            }
        }
        sum = add_cd(sum, mul_cd(value, x[colIdx[p]]));
    }
    y[row] = sum;
}

// Applies an on-the-fly affine Liouvillian to a batch of vectors
__global__ void fused_affine_spmv_batch_kernel(int nrows, int nvec, int nnz, int nctrls,
                                               int driftIndex,
                                               const int* rowPtr, const int* colIdx,
                                               const cdouble* driftValues,
                                               const cdouble* controlValues,
                                               const double* coeff,
                                               const cdouble* xBatch, cdouble* yBatch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nrows * nvec;
    if (idx >= total) return;
    int row = idx % nrows;
    int vec = idx / nrows;
    const cdouble* x = xBatch + static_cast<size_t>(vec) * nrows;
    cdouble* y = yBatch + static_cast<size_t>(vec) * nrows;
    const size_t driftBase = static_cast<size_t>(driftIndex) * nnz;
    cdouble sum = make_cd(0.0, 0.0);
    for (int p = rowPtr[row]; p < rowPtr[row+1]; ++p) {
        cdouble value = driftValues[driftBase + p];
        for (int k = 0; k < nctrls; ++k) {
            double c = coeff[k];
            if (c != 0.0) {
                value = add_cd(value, scale_cd(controlValues[static_cast<size_t>(k) * nnz + p], c));
            }
        }
        sum = add_cd(sum, mul_cd(value, x[colIdx[p]]));
    }
    y[row] = sum;
}

// Applies a block of control operators from the fused CSR to a batch of vectors
__global__ void fused_control_spmv_batch_kernel(int nrows, int nvec, int nnz, int firstCtrl,
                                                const int* rowPtr, const int* colIdx,
                                                const cdouble* controlValues,
                                                const cdouble* x, cdouble* yBatch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nrows * nvec;
    if (idx >= total) return;
    int row = idx % nrows;
    int vec = idx / nrows;
    cdouble* y = yBatch + static_cast<size_t>(vec) * nrows;
    const size_t ctrlBase = static_cast<size_t>(firstCtrl + vec) * nnz;
    cdouble sum = make_cd(0.0, 0.0);
    for (int p = rowPtr[row]; p < rowPtr[row+1]; ++p) {
        sum = add_cd(sum, mul_cd(controlValues[ctrlBase + p], x[colIdx[p]]));
    }
    y[row] = sum;
}

// Advances the coupled state and derivative recurrence in one fused pass
__global__ void fused_implicit_frechet_batch_kernel(int nrows, int nvec, int nnz,
                                                    int nctrls, int driftIndex,
                                                    int firstCtrl,
                                                    const int* rowPtr,
                                                    const int* colIdx,
                                                    const cdouble* driftValues,
                                                    const cdouble* controlValues,
                                                    const double* coeff,
                                                    const cdouble* etaBatch,
                                                    const cdouble* rho,
                                                    cdouble* outEtaBatch,
                                                    cdouble* outRho) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nrows * nvec;
    if (idx >= total) return;
    int row = idx % nrows;
    int vec = idx / nrows;
    const cdouble* eta = etaBatch + static_cast<size_t>(vec) * nrows;
    cdouble* outEta = outEtaBatch + static_cast<size_t>(vec) * nrows;
    const size_t driftBase = static_cast<size_t>(driftIndex) * nnz;
    const size_t ctrlBase = static_cast<size_t>(firstCtrl + vec) * nnz;

    cdouble etaSum = make_cd(0.0, 0.0);
    cdouble rhoSum = make_cd(0.0, 0.0);
    for (int p = rowPtr[row]; p < rowPtr[row+1]; ++p) {
        int col = colIdx[p];
        cdouble a = driftValues[driftBase + p];
        for (int k = 0; k < nctrls; ++k) {
            double c = coeff[k];
            if (c != 0.0) {
                a = add_cd(a, scale_cd(controlValues[static_cast<size_t>(k) * nnz + p], c));
            }
        }
        cdouble rhoVal = rho[col];
        etaSum = add_cd(etaSum, mul_cd(a, eta[col]));
        etaSum = add_cd(etaSum, mul_cd(controlValues[ctrlBase + p], rhoVal));
        if (vec == 0) rhoSum = add_cd(rhoSum, mul_cd(a, rhoVal));
    }
    outEta[row] = etaSum;
    if (vec == 0) outRho[row] = rhoSum;
}

// Applies a per-member affine Liouvillian across the whole ensemble
__global__ void fused_affine_ensemble_spmv_kernel(int nrows, int nsystems, int nnz,
                                                  int nctrls, int ndrifts, int driftIndex,
                                                  const int* rowPtr, const int* colIdx,
                                                  const cdouble* driftValues,
                                                  const cdouble* controlValues,
                                                  const double* coeff,
                                                  const cdouble* xBatch, cdouble* yBatch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nrows * nsystems;
    if (idx >= total) return;
    int row = idx % nrows;
    int sys = idx / nrows;
    const cdouble* x = xBatch + static_cast<size_t>(sys) * nrows;
    cdouble* y = yBatch + static_cast<size_t>(sys) * nrows;
    const size_t driftBase = (static_cast<size_t>(sys) * ndrifts + driftIndex) * nnz;
    cdouble sum = make_cd(0.0, 0.0);
    for (int p = rowPtr[row]; p < rowPtr[row+1]; ++p) {
        cdouble value = driftValues[driftBase + p];
        for (int k = 0; k < nctrls; ++k) {
            double c = coeff[k];
            if (c != 0.0) {
                value = add_cd(value, scale_cd(controlValues[static_cast<size_t>(k) * nnz + p], c));
            }
        }
        sum = add_cd(sum, mul_cd(value, x[colIdx[p]]));
    }
    y[row] = sum;
}

// Applies a per-member affine Liouvillian to a batch of vectors per member
__global__ void fused_affine_ensemble_spmv_batch_kernel(int nrows, int nsystems,
                                                        int nvecPerSystem, int nnz,
                                                        int nctrls, int ndrifts,
                                                        int driftIndex,
                                                        const int* rowPtr, const int* colIdx,
                                                        const cdouble* driftValues,
                                                        const cdouble* controlValues,
                                                        const double* coeff,
                                                        const cdouble* xBatch,
                                                        cdouble* yBatch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalVec = nsystems * nvecPerSystem;
    int total = nrows * totalVec;
    if (idx >= total) return;
    int row = idx % nrows;
    int vec = idx / nrows;
    int sys = vec / nvecPerSystem;
    const cdouble* x = xBatch + static_cast<size_t>(vec) * nrows;
    cdouble* y = yBatch + static_cast<size_t>(vec) * nrows;
    const size_t driftBase = (static_cast<size_t>(sys) * ndrifts + driftIndex) * nnz;
    cdouble sum = make_cd(0.0, 0.0);
    for (int p = rowPtr[row]; p < rowPtr[row+1]; ++p) {
        cdouble value = driftValues[driftBase + p];
        for (int k = 0; k < nctrls; ++k) {
            double c = coeff[k];
            if (c != 0.0) {
                value = add_cd(value, scale_cd(controlValues[static_cast<size_t>(k) * nnz + p], c));
            }
        }
        sum = add_cd(sum, mul_cd(value, x[colIdx[p]]));
    }
    y[row] = sum;
}

// Applies a block of control operators to a batch of vectors per ensemble member
__global__ void fused_control_ensemble_spmv_batch_kernel(int nrows, int nsystems,
                                                         int nvecPerSystem, int nnz,
                                                         int firstCtrl,
                                                         const int* rowPtr, const int* colIdx,
                                                         const cdouble* controlValues,
                                                         const cdouble* xSystems,
                                                         cdouble* yBatch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalVec = nsystems * nvecPerSystem;
    int total = nrows * totalVec;
    if (idx >= total) return;
    int row = idx % nrows;
    int vec = idx / nrows;
    int sys = vec / nvecPerSystem;
    int localCtrl = vec - sys * nvecPerSystem;
    const cdouble* x = xSystems + static_cast<size_t>(sys) * nrows;
    cdouble* y = yBatch + static_cast<size_t>(vec) * nrows;
    const size_t ctrlBase = static_cast<size_t>(firstCtrl + localCtrl) * nnz;
    cdouble sum = make_cd(0.0, 0.0);
    for (int p = rowPtr[row]; p < rowPtr[row+1]; ++p) {
        sum = add_cd(sum, mul_cd(controlValues[ctrlBase + p], x[colIdx[p]]));
    }
    y[row] = sum;
}

// Assembles the affine Liouvillian of every ensemble member into CSR storage
__global__ void assemble_affine_ensemble_kernel(int nsystems, int nnz, int nctrls,
                                                int ndrifts, int driftIndex,
                                                const cdouble* driftValues,
                                                const cdouble* controlValues,
                                                const double* coeff,
                                                cdouble* affineValues) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nsystems * nnz;
    if (idx >= total) return;
    int p = idx % nnz;
    int sys = idx / nnz;
    cdouble value = driftValues[(static_cast<size_t>(sys) * ndrifts + driftIndex) * nnz + p];
    for (int k = 0; k < nctrls; ++k) {
        double c = coeff[k];
        if (c != 0.0) value = add_cd(value, scale_cd(controlValues[static_cast<size_t>(k) * nnz + p], c));
    }
    affineValues[idx] = value;
}

// Assembles the affine Liouvillian of every ensemble member into SELL-32 storage
__global__ void assemble_affine_ensemble_sell_kernel(int nsystems, int sellNnz,
                                                     int nctrls, int ndrifts,
                                                     int driftIndex,
                                                     const cdouble* driftValues,
                                                     const cdouble* controlValues,
                                                     const double* coeff,
                                                     cdouble* affineValues) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nsystems * sellNnz;
    if (idx >= total) return;
    int p = idx % sellNnz;
    int sys = idx / sellNnz;
    cdouble value = driftValues[(static_cast<size_t>(sys) * ndrifts + driftIndex) * sellNnz + p];
    for (int k = 0; k < nctrls; ++k) {
        double c = coeff[k];
        if (c != 0.0) value = add_cd(value, scale_cd(controlValues[static_cast<size_t>(k) * sellNnz + p], c));
    }
    affineValues[idx] = value;
}

// Applies preassembled ensemble affine values held in CSR storage
__global__ void affine_ensemble_spmv_kernel(int nrows, int nsystems, int nnz,
                                            const int* rowPtr, const int* colIdx,
                                            const cdouble* affineValues,
                                            const cdouble* xBatch,
                                            cdouble* yBatch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nrows * nsystems;
    if (idx >= total) return;
    int row = idx % nrows;
    int sys = idx / nrows;
    const cdouble* x = xBatch + static_cast<size_t>(sys) * nrows;
    cdouble* y = yBatch + static_cast<size_t>(sys) * nrows;
    const cdouble* values = affineValues + static_cast<size_t>(sys) * nnz;
    cdouble sum = make_cd(0.0, 0.0);
    for (int p = rowPtr[row]; p < rowPtr[row+1]; ++p) {
        sum = add_cd(sum, mul_cd(values[p], x[colIdx[p]]));
    }
    y[row] = sum;
}

// Applies preassembled ensemble affine values in CSR storage to a batch of vectors
__global__ void affine_ensemble_spmv_batch_kernel(int nrows, int nsystems,
                                                  int nvecPerSystem, int nnz,
                                                  const int* rowPtr, const int* colIdx,
                                                  const cdouble* affineValues,
                                                  const cdouble* xBatch,
                                                  cdouble* yBatch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalVec = nsystems * nvecPerSystem;
    int total = nrows * totalVec;
    if (idx >= total) return;
    int row = idx % nrows;
    int vec = idx / nrows;
    int sys = vec / nvecPerSystem;
    const cdouble* x = xBatch + static_cast<size_t>(vec) * nrows;
    cdouble* y = yBatch + static_cast<size_t>(vec) * nrows;
    const cdouble* values = affineValues + static_cast<size_t>(sys) * nnz;
    cdouble sum = make_cd(0.0, 0.0);
    for (int p = rowPtr[row]; p < rowPtr[row+1]; ++p) {
        sum = add_cd(sum, mul_cd(values[p], x[colIdx[p]]));
    }
    y[row] = sum;
}

// Applies preassembled ensemble affine values held in SELL-32 storage
__global__ void affine_ensemble_spmv_sell_kernel(int nrows, int nsystems,
                                                 int sellNnz,
                                                 const int* slicePtr,
                                                 const int* sliceWidth,
                                                 const int* sellColIdx,
                                                 const cdouble* affineValues,
                                                 const cdouble* xBatch,
                                                 cdouble* yBatch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nrows * nsystems;
    if (idx >= total) return;
    int row = idx % nrows;
    int sys = idx / nrows;
    int slice = row / SELL_SLICE_HEIGHT;
    int local = row - slice * SELL_SLICE_HEIGHT;
    int width = sliceWidth[slice];
    int base = slicePtr[slice] + local;
    const cdouble* values = affineValues + static_cast<size_t>(sys) * sellNnz;
    const cdouble* x = xBatch + static_cast<size_t>(sys) * nrows;
    cdouble sum = make_cd(0.0, 0.0);
    for (int slot = 0; slot < width; ++slot) {
        int p = base + slot * SELL_SLICE_HEIGHT;
        int col = sellColIdx[p];
        if (col >= 0) sum = add_cd(sum, mul_cd(values[p], x[col]));
    }
    yBatch[static_cast<size_t>(sys) * nrows + row] = sum;
}

// Applies preassembled ensemble affine values in SELL-32 storage to a batch of vectors
__global__ void affine_ensemble_spmv_batch_sell_kernel(int nrows, int nsystems,
                                                       int nvecPerSystem,
                                                       int sellNnz,
                                                       const int* slicePtr,
                                                       const int* sliceWidth,
                                                       const int* sellColIdx,
                                                       const cdouble* affineValues,
                                                       const cdouble* xBatch,
                                                       cdouble* yBatch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalVec = nsystems * nvecPerSystem;
    int total = nrows * totalVec;
    if (idx >= total) return;
    int row = idx % nrows;
    int vec = idx / nrows;
    int sys = vec / nvecPerSystem;
    int slice = row / SELL_SLICE_HEIGHT;
    int local = row - slice * SELL_SLICE_HEIGHT;
    int width = sliceWidth[slice];
    int base = slicePtr[slice] + local;
    const cdouble* values = affineValues + static_cast<size_t>(sys) * sellNnz;
    const cdouble* x = xBatch + static_cast<size_t>(vec) * nrows;
    cdouble sum = make_cd(0.0, 0.0);
    for (int slot = 0; slot < width; ++slot) {
        int p = base + slot * SELL_SLICE_HEIGHT;
        int col = sellColIdx[p];
        if (col >= 0) sum = add_cd(sum, mul_cd(values[p], x[col]));
    }
    yBatch[static_cast<size_t>(vec) * nrows + row] = sum;
}

// Applies preassembled ensemble affine values with one half-warp per row
__global__ void affine_ensemble_spmv_halfwarp_kernel(int nrows, int nsystems,
                                                     int nnz,
                                                     const int* rowPtr,
                                                     const int* colIdx,
                                                     const cdouble* affineValues,
                                                     const cdouble* xBatch,
                                                     cdouble* yBatch) {
    int group = blockIdx.x * (blockDim.x / 16) + threadIdx.x / 16;
    int total = nrows * nsystems;
    if (group >= total) return;
    int lane = threadIdx.x & 15;
    int row = group % nrows;
    int sys = group / nrows;
    const cdouble* x = xBatch + static_cast<size_t>(sys) * nrows;
    cdouble* y = yBatch + static_cast<size_t>(sys) * nrows;
    const cdouble* values = affineValues + static_cast<size_t>(sys) * nnz;
    cdouble sum = make_cd(0.0, 0.0);
    for (int p = rowPtr[row] + lane; p < rowPtr[row+1]; p += 16) {
        sum = add_cd(sum, mul_cd(values[p], x[colIdx[p]]));
    }
    unsigned mask = (threadIdx.x & 16) ? 0xffff0000u : 0x0000ffffu;
    sum = halfWarpSum(sum, mask);
    if (lane == 0) y[row] = sum;
}

// Applies preassembled ensemble affine values to a batch, one half-warp per row
__global__ void affine_ensemble_spmv_batch_halfwarp_kernel(int nrows, int nsystems,
                                                           int nvecPerSystem,
                                                           int nnz,
                                                           const int* rowPtr,
                                                           const int* colIdx,
                                                           const cdouble* affineValues,
                                                           const cdouble* xBatch,
                                                           cdouble* yBatch) {
    int group = blockIdx.x * (blockDim.x / 16) + threadIdx.x / 16;
    int totalVec = nsystems * nvecPerSystem;
    int total = nrows * totalVec;
    if (group >= total) return;
    int lane = threadIdx.x & 15;
    int row = group % nrows;
    int vec = group / nrows;
    int sys = vec / nvecPerSystem;
    const cdouble* x = xBatch + static_cast<size_t>(vec) * nrows;
    cdouble* y = yBatch + static_cast<size_t>(vec) * nrows;
    const cdouble* values = affineValues + static_cast<size_t>(sys) * nnz;
    cdouble sum = make_cd(0.0, 0.0);
    for (int p = rowPtr[row] + lane; p < rowPtr[row+1]; p += 16) {
        sum = add_cd(sum, mul_cd(values[p], x[colIdx[p]]));
    }
    unsigned mask = (threadIdx.x & 16) ? 0xffff0000u : 0x0000ffffu;
    sum = halfWarpSum(sum, mask);
    if (lane == 0) y[row] = sum;
}

// Advances the coupled ensemble state and derivative recurrence in CSR storage
__global__ void implicit_frechet_ensemble_affine_batch_kernel(int nrows, int nsystems,
                                                              int nvecPerSystem,
                                                              int nnz, int firstCtrl,
                                                              const int* rowPtr,
                                                              const int* colIdx,
                                                              const cdouble* affineValues,
                                                              const cdouble* controlValues,
                                                              const cdouble* etaBatch,
                                                              const cdouble* rhoSystems,
                                                              cdouble* outEtaBatch,
                                                              cdouble* outRhoSystems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalVec = nsystems * nvecPerSystem;
    int total = nrows * totalVec;
    if (idx >= total) return;
    int row = idx % nrows;
    int vec = idx / nrows;
    int sys = vec / nvecPerSystem;
    int localCtrl = vec - sys * nvecPerSystem;

    const cdouble* eta = etaBatch + static_cast<size_t>(vec) * nrows;
    cdouble* outEta = outEtaBatch + static_cast<size_t>(vec) * nrows;
    const cdouble* rho = rhoSystems + static_cast<size_t>(sys) * nrows;
    cdouble* outRho = outRhoSystems + static_cast<size_t>(sys) * nrows;
    const cdouble* values = affineValues + static_cast<size_t>(sys) * nnz;
    const size_t ctrlBase = static_cast<size_t>(firstCtrl + localCtrl) * nnz;

    cdouble etaSum = make_cd(0.0, 0.0);
    cdouble rhoSum = make_cd(0.0, 0.0);
    for (int p = rowPtr[row]; p < rowPtr[row+1]; ++p) {
        int col = colIdx[p];
        cdouble a = values[p];
        cdouble rhoVal = rho[col];
        etaSum = add_cd(etaSum, mul_cd(a, eta[col]));
        etaSum = add_cd(etaSum, mul_cd(controlValues[ctrlBase + p], rhoVal));
        if (localCtrl == 0) rhoSum = add_cd(rhoSum, mul_cd(a, rhoVal));
    }
    outEta[row] = etaSum;
    if (localCtrl == 0) outRho[row] = rhoSum;
}

// Advances the coupled ensemble state and derivative recurrence in SELL-32 storage
__global__ void implicit_frechet_ensemble_affine_batch_sell_kernel(int nrows,
                                                                   int nsystems,
                                                                   int nvecPerSystem,
                                                                   int sellNnz,
                                                                   int firstCtrl,
                                                                   const int* slicePtr,
                                                                   const int* sliceWidth,
                                                                   const int* sellColIdx,
                                                                   const cdouble* affineValues,
                                                                   const cdouble* controlValues,
                                                                   const cdouble* etaBatch,
                                                                   const cdouble* rhoSystems,
                                                                   cdouble* outEtaBatch,
                                                                   cdouble* outRhoSystems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalVec = nsystems * nvecPerSystem;
    int total = nrows * totalVec;
    if (idx >= total) return;
    int row = idx % nrows;
    int vec = idx / nrows;
    int sys = vec / nvecPerSystem;
    int localCtrl = vec - sys * nvecPerSystem;
    int slice = row / SELL_SLICE_HEIGHT;
    int local = row - slice * SELL_SLICE_HEIGHT;
    int width = sliceWidth[slice];
    int base = slicePtr[slice] + local;

    const cdouble* eta = etaBatch + static_cast<size_t>(vec) * nrows;
    cdouble* outEta = outEtaBatch + static_cast<size_t>(vec) * nrows;
    const cdouble* rho = rhoSystems + static_cast<size_t>(sys) * nrows;
    cdouble* outRho = outRhoSystems + static_cast<size_t>(sys) * nrows;
    const cdouble* values = affineValues + static_cast<size_t>(sys) * sellNnz;
    const size_t ctrlBase = static_cast<size_t>(firstCtrl + localCtrl) * sellNnz;

    cdouble etaSum = make_cd(0.0, 0.0);
    cdouble rhoSum = make_cd(0.0, 0.0);
    for (int slot = 0; slot < width; ++slot) {
        int p = base + slot * SELL_SLICE_HEIGHT;
        int col = sellColIdx[p];
        if (col < 0) continue;
        cdouble a = values[p];
        cdouble rhoVal = rho[col];
        etaSum = add_cd(etaSum, mul_cd(a, eta[col]));
        etaSum = add_cd(etaSum, mul_cd(controlValues[ctrlBase + p], rhoVal));
        if (localCtrl == 0) rhoSum = add_cd(rhoSum, mul_cd(a, rhoVal));
    }
    outEta[row] = etaSum;
    if (localCtrl == 0) outRho[row] = rhoSum;
}

// Advances the coupled ensemble recurrence with one half-warp per row
__global__ void implicit_frechet_ensemble_affine_batch_halfwarp_kernel(int nrows,
                                                                       int nsystems,
                                                                       int nvecPerSystem,
                                                                       int nnz,
                                                                       int firstCtrl,
                                                                       const int* rowPtr,
                                                                       const int* colIdx,
                                                                       const cdouble* affineValues,
                                                                       const cdouble* controlValues,
                                                                       const cdouble* etaBatch,
                                                                       const cdouble* rhoSystems,
                                                                       cdouble* outEtaBatch,
                                                                       cdouble* outRhoSystems) {
    int group = blockIdx.x * (blockDim.x / 16) + threadIdx.x / 16;
    int totalVec = nsystems * nvecPerSystem;
    int total = nrows * totalVec;
    if (group >= total) return;
    int lane = threadIdx.x & 15;
    int row = group % nrows;
    int vec = group / nrows;
    int sys = vec / nvecPerSystem;
    int localCtrl = vec - sys * nvecPerSystem;

    const cdouble* eta = etaBatch + static_cast<size_t>(vec) * nrows;
    cdouble* outEta = outEtaBatch + static_cast<size_t>(vec) * nrows;
    const cdouble* rho = rhoSystems + static_cast<size_t>(sys) * nrows;
    cdouble* outRho = outRhoSystems + static_cast<size_t>(sys) * nrows;
    const cdouble* values = affineValues + static_cast<size_t>(sys) * nnz;
    const size_t ctrlBase = static_cast<size_t>(firstCtrl + localCtrl) * nnz;

    cdouble etaSum = make_cd(0.0, 0.0);
    cdouble rhoSum = make_cd(0.0, 0.0);
    for (int p = rowPtr[row] + lane; p < rowPtr[row+1]; p += 16) {
        int col = colIdx[p];
        cdouble a = values[p];
        cdouble rhoVal = rho[col];
        etaSum = add_cd(etaSum, mul_cd(a, eta[col]));
        etaSum = add_cd(etaSum, mul_cd(controlValues[ctrlBase + p], rhoVal));
        if (localCtrl == 0) rhoSum = add_cd(rhoSum, mul_cd(a, rhoVal));
    }
    unsigned mask = (threadIdx.x & 16) ? 0xffff0000u : 0x0000ffffu;
    etaSum = halfWarpSum(etaSum, mask);
    if (localCtrl == 0) rhoSum = halfWarpSum(rhoSum, mask);
    if (lane == 0) {
        outEta[row] = etaSum;
        if (localCtrl == 0) outRho[row] = rhoSum;
    }
}

// Advances the coupled ensemble recurrence with the affine values assembled on the fly
__global__ void fused_implicit_frechet_ensemble_batch_kernel(int nrows, int nsystems,
                                                             int nvecPerSystem,
                                                             int nnz, int nctrls,
                                                             int ndrifts,
                                                             int driftIndex,
                                                             int firstCtrl,
                                                             const int* rowPtr,
                                                             const int* colIdx,
                                                             const cdouble* driftValues,
                                                             const cdouble* controlValues,
                                                             const double* coeff,
                                                             const cdouble* etaBatch,
                                                             const cdouble* rhoSystems,
                                                             cdouble* outEtaBatch,
                                                             cdouble* outRhoSystems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalVec = nsystems * nvecPerSystem;
    int total = nrows * totalVec;
    if (idx >= total) return;
    int row = idx % nrows;
    int vec = idx / nrows;
    int sys = vec / nvecPerSystem;
    int localCtrl = vec - sys * nvecPerSystem;

    const cdouble* eta = etaBatch + static_cast<size_t>(vec) * nrows;
    cdouble* outEta = outEtaBatch + static_cast<size_t>(vec) * nrows;
    const cdouble* rho = rhoSystems + static_cast<size_t>(sys) * nrows;
    cdouble* outRho = outRhoSystems + static_cast<size_t>(sys) * nrows;
    const size_t driftBase = (static_cast<size_t>(sys) * ndrifts + driftIndex) * nnz;
    const size_t ctrlBase = static_cast<size_t>(firstCtrl + localCtrl) * nnz;

    cdouble etaSum = make_cd(0.0, 0.0);
    cdouble rhoSum = make_cd(0.0, 0.0);
    for (int p = rowPtr[row]; p < rowPtr[row+1]; ++p) {
        int col = colIdx[p];
        cdouble a = driftValues[driftBase + p];
        for (int k = 0; k < nctrls; ++k) {
            double c = coeff[k];
            if (c != 0.0) {
                a = add_cd(a, scale_cd(controlValues[static_cast<size_t>(k) * nnz + p], c));
            }
        }
        cdouble rhoVal = rho[col];
        etaSum = add_cd(etaSum, mul_cd(a, eta[col]));
        etaSum = add_cd(etaSum, mul_cd(controlValues[ctrlBase + p], rhoVal));
        if (localCtrl == 0) rhoSum = add_cd(rhoSum, mul_cd(a, rhoVal));
    }
    outEta[row] = etaSum;
    if (localCtrl == 0) outRho[row] = rhoSum;
}

// Reduces the largest absolute value of a complex vector, one result per block
__global__ void max_abs_reduce_kernel(const cdouble* x, int n, double* blockMax) {
    __shared__ double sdata[THREADS];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double m = 0.0;
    while (i < n) {
        m = fmax(m, hypot(x[i].x, x[i].y));
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = m;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmax(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) blockMax[blockIdx.x] = sdata[0];
}

// Reduces the largest absolute value across two complex vectors
__global__ void max_abs_pair_reduce_kernel(const cdouble* a, int na,
                                           const cdouble* b, int nb,
                                           double* blockMax) {
    __shared__ double sdata[THREADS];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = max(na, nb);
    double m = 0.0;
    while (i < n) {
        if (i < na) m = fmax(m, hypot(a[i].x, a[i].y));
        if (i < nb) m = fmax(m, hypot(b[i].x, b[i].y));
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = m;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmax(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) blockMax[blockIdx.x] = sdata[0];
}

// Computes one inner product and stores it at the requested output index
__global__ void dot_store_kernel(const cdouble* a, const cdouble* b, int n,
                                 cdouble* out, int outIndex) {
    __shared__ cdouble sdata[THREADS];
    int tid = threadIdx.x;
    int i = threadIdx.x;
    cdouble sum = make_cd(0.0, 0.0);
    while (i < n) {
        sum = add_cd(sum, mul_cd(conj_cd(a[i]), b[i]));
        i += blockDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = add_cd(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) out[outIndex] = sdata[0];
}

// Computes the inner products of one vector with a batch of vectors
__global__ void dot_batch_store_kernel(const cdouble* a, const cdouble* bBatch,
                                       int n, int nvec, cdouble* out,
                                       int outStart, int outStride) {
    __shared__ cdouble sdata[THREADS];
    int vec = blockIdx.x;
    if (vec >= nvec) return;
    const cdouble* b = bBatch + static_cast<size_t>(vec) * n;
    int tid = threadIdx.x;
    int i = threadIdx.x;
    cdouble sum = make_cd(0.0, 0.0);
    while (i < n) {
        sum = add_cd(sum, mul_cd(conj_cd(a[i]), b[i]));
        i += blockDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = add_cd(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) out[outStart + vec*outStride] = sdata[0];
}

// Computes one inner product per ensemble member
__global__ void dot_ensemble_store_kernel(const cdouble* aBatch, const cdouble* bBatch,
                                          int n, int nsystems, cdouble* out) {
    __shared__ cdouble sdata[THREADS];
    int sys = blockIdx.x;
    if (sys >= nsystems) return;
    const cdouble* a = aBatch + static_cast<size_t>(sys) * n;
    const cdouble* b = bBatch + static_cast<size_t>(sys) * n;
    int tid = threadIdx.x;
    int i = threadIdx.x;
    cdouble sum = make_cd(0.0, 0.0);
    while (i < n) {
        sum = add_cd(sum, mul_cd(conj_cd(a[i]), b[i]));
        i += blockDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = add_cd(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) out[sys] = sdata[0];
}

// Computes the inner products of a batch of vectors for every ensemble member
__global__ void dot_ensemble_batch_store_kernel(const cdouble* aSystems,
                                                const cdouble* bBatch,
                                                int n, int nsystems,
                                                int nvecPerSystem,
                                                cdouble* out,
                                                int outStart) {
    __shared__ cdouble sdata[THREADS];
    int vec = blockIdx.x;
    int totalVec = nsystems * nvecPerSystem;
    if (vec >= totalVec) return;
    int sys = vec / nvecPerSystem;
    int local = vec - sys * nvecPerSystem;
    const cdouble* a = aSystems + static_cast<size_t>(sys) * n;
    const cdouble* b = bBatch + static_cast<size_t>(vec) * n;
    int tid = threadIdx.x;
    int i = threadIdx.x;
    cdouble sum = make_cd(0.0, 0.0);
    while (i < n) {
        sum = add_cd(sum, mul_cd(conj_cd(a[i]), b[i]));
        i += blockDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = add_cd(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) out[static_cast<size_t>(outStart + local) * nsystems + sys] = sdata[0];
}

// Turns the raw overlap derivatives into the requested fidelity gradient
__global__ void postprocess_gradient_kernel(cdouble* grad, int n, cdouble overlap, int mode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    cdouble g = grad[i];
    if (mode == 0) {             // real
        grad[i] = make_cd(g.x, 0.0);
    } else if (mode == 1) {      // imag
        grad[i] = make_cd(g.y, 0.0);
    } else {                     // square: g*conj(overlap)+overlap*conj(g), real
        cdouble term = add_cd(mul_cd(g, conj_cd(overlap)), mul_cd(overlap, conj_cd(g)));
        grad[i] = make_cd(term.x, 0.0);
    }
}

// Turns the raw overlap derivatives into one fidelity gradient per ensemble member
__global__ void postprocess_ensemble_gradient_kernel(cdouble* grad, int ngrad,
                                                     int nsystems, const cdouble* overlap,
                                                     int mode) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ngrad * nsystems;
    if (idx >= total) return;
    int sys = idx % nsystems;
    cdouble g = grad[idx];
    cdouble ov = overlap[sys];
    if (mode == 0) {
        grad[idx] = make_cd(g.x, 0.0);
    } else if (mode == 1) {
        grad[idx] = make_cd(g.y, 0.0);
    } else {
        cdouble term = add_cd(mul_cd(g, conj_cd(ov)), mul_cd(ov, conj_cd(g)));
        grad[idx] = make_cd(term.x, 0.0);
    }
}

// Averages the real part of the ensemble gradients into the output gradient
__global__ void average_real_gradient_kernel(const cdouble* gradEns, double* gradOut,
                                             int ngrad, int nsystems) {
    __shared__ double sdata[THREADS];
    int g = blockIdx.x;
    if (g >= ngrad) return;
    int tid = threadIdx.x;
    double sum = 0.0;
    for (int sys = tid; sys < nsystems; sys += blockDim.x) {
        sum += gradEns[static_cast<size_t>(g) * nsystems + sys].x;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) gradOut[g] = sdata[0] / static_cast<double>(nsystems);
}

}  // namespace

#endif  // GRAPE_GPU_KERNELS_CUH
