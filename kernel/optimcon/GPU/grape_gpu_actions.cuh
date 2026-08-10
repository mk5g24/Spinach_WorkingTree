/*
 * grape_gpu_actions.cuh
 *
 * Launchers.  One function here is one application of an operator to one or
 * more vectors, with the launch geometry worked out and the work counted.
 *
 * The layers are: launch-size arithmetic, thin wrappers around the
 * elementwise kernels, affine Liouvillian actions in scalar and ensemble
 * form, and the fused reductions that copy a vector and return its norm in
 * the same pass.  Nothing here advances time; the recurrences that do live
 * in grape_gpu_propagate.cuh.
 */

#ifndef GRAPE_GPU_ACTIONS_CUH
#define GRAPE_GPU_ACTIONS_CUH

#include "grape_gpu_kernels.cuh"
#include "grape_gpu_stats.cuh"

namespace {

// Number of thread blocks needed to cover n elements
int blocksFor(int n) {
    return std::max(1, (n + THREADS - 1) / THREADS);
}

// Number of thread blocks needed to cover a number of half-warp groups
int blocksForHalfWarpGroups(int nGroups) {
    constexpr int GROUPS_PER_BLOCK = THREADS / 16;
    return std::max(1, (nGroups + GROUPS_PER_BLOCK - 1) / GROUPS_PER_BLOCK);
}

// Element count of a batch, refusing sizes that overflow 32-bit launch indexing
int checkedBatchElements(int n, int nvec, const char* what) {
    size_t total = static_cast<size_t>(n) * static_cast<size_t>(nvec);
    if (total > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(what) + " exceeds 32-bit CUDA launch indexing in this backend.");
    }
    return static_cast<int>(total);
}

// Copies a complex vector on the device
void copyDevice(const cdouble* src, cdouble* dst, int n) {
    copy_kernel<<<blocksFor(n), THREADS>>>(src, dst, n);
    cudaCheck(cudaGetLastError(), "copy_kernel");
}

// Fills a complex vector on the device with a constant
void fillDevice(cdouble* dst, int n, cdouble value = make_cd(0.0, 0.0)) {
    fill_kernel<<<blocksFor(n), THREADS>>>(dst, n, value);
    cudaCheck(cudaGetLastError(), "fill_kernel");
}

// Scales a complex vector on the device by a real number
void scaleDevice(cdouble* dst, int n, double s) {
    scale_kernel<<<blocksFor(n), THREADS>>>(dst, n, s);
    cudaCheck(cudaGetLastError(), "scale_kernel");
}

// Adds one device vector into another
void axpyDevice(cdouble* y, const cdouble* x, int n) {
    axpy_kernel<<<blocksFor(n), THREADS>>>(y, x, n);
    cudaCheck(cudaGetLastError(), "axpy_kernel");
}

// Scales a device vector by a complex number and accumulates the result
void linearScaleAxpyDevice(cdouble* y, const cdouble* x, cdouble* accum,
                           int n, cdouble alpha, const char* where) {
    linear_scale_axpy_kernel<<<blocksFor(n), THREADS>>>(y, x, accum, n, alpha);
    cudaCheck(cudaGetLastError(), where);
}

// Accumulates a scaled sparse matrix-vector product
void spmvAccum(const CsrDevice& A, double alpha, const cdouble* x, cdouble* y) {
    spmv_accum_kernel<<<blocksFor(A.n), THREADS>>>(A.n, A.rowPtr, A.colIdx,
                                                   A.values, alpha, x, y);
    cudaCheck(cudaGetLastError(), "spmv_accum_kernel");
}

// Accumulates a scaled sparse product for a batch of vectors sharing one matrix
void spmvAccumBatchSameMatrix(const CsrDevice& A, double alpha, const cdouble* x,
                              cdouble* y, int nvec) {
    int total = checkedBatchElements(A.n, nvec, "batched sparse matrix-vector input");
    spmv_accum_batch_same_matrix_kernel<<<blocksFor(total), THREADS>>>(
        A.n, nvec, A.rowPtr, A.colIdx, A.values, alpha, x, y);
    cudaCheck(cudaGetLastError(), "spmv_accum_batch_same_matrix_kernel");
}

// Applies an affine Liouvillian assembled on the fly from the fused CSR
void fusedAffineAction(const FusedCsrDevice& F, int driftIndex, const double* dCoeff,
                       const cdouble* x, cdouble* y) {
    fused_affine_spmv_kernel<<<blocksFor(F.n), THREADS>>>(
        F.n, F.nnz, F.nctrls, driftIndex, F.rowPtr, F.colIdx,
        F.driftValues, F.controlValues, dCoeff, x, y);
    cudaCheck(cudaGetLastError(), "fused_affine_spmv_kernel");
}

// Applies an on-the-fly affine Liouvillian to a batch of vectors
void fusedAffineActionBatch(const FusedCsrDevice& F, int driftIndex, const double* dCoeff,
                            const cdouble* xBatch, cdouble* yBatch, int nvec) {
    int total = checkedBatchElements(F.n, nvec, "batched fused affine action");
    fused_affine_spmv_batch_kernel<<<blocksFor(total), THREADS>>>(
        F.n, nvec, F.nnz, F.nctrls, driftIndex, F.rowPtr, F.colIdx,
        F.driftValues, F.controlValues, dCoeff, xBatch, yBatch);
    cudaCheck(cudaGetLastError(), "fused_affine_spmv_batch_kernel");
}

// Applies a block of control operators to a batch of vectors
void fusedControlActionBatch(const FusedCsrDevice& F, int firstCtrl, int nvec,
                             const cdouble* x, cdouble* yBatch) {
    int total = checkedBatchElements(F.n, nvec, "batched fused control action");
    fused_control_spmv_batch_kernel<<<blocksFor(total), THREADS>>>(
        F.n, nvec, F.nnz, firstCtrl, F.rowPtr, F.colIdx, F.controlValues, x, yBatch);
    cudaCheck(cudaGetLastError(), "fused_control_spmv_batch_kernel");
}

// Advances the coupled state and derivative recurrence in one fused pass
void fusedImplicitFrechetActionBatch(const FusedCsrDevice& F, int driftIndex,
                                     const double* dCoeff, int firstCtrl, int nvec,
                                     const cdouble* etaBatch, const cdouble* rho,
                                     cdouble* outEtaBatch, cdouble* outRho) {
    int total = checkedBatchElements(F.n, nvec, "implicit Frechet fused batch action");
    fused_implicit_frechet_batch_kernel<<<blocksFor(total), THREADS>>>(
        F.n, nvec, F.nnz, F.nctrls, driftIndex, firstCtrl,
        F.rowPtr, F.colIdx, F.driftValues, F.controlValues, dCoeff,
        etaBatch, rho, outEtaBatch, outRho);
    cudaCheck(cudaGetLastError(), "fused_implicit_frechet_batch_kernel");
}

// Applies an on-the-fly affine Liouvillian across the whole ensemble
void fusedAffineActionEnsemble(const FusedCsrDevice& F, int driftIndex, const double* dCoeff,
                               const cdouble* xSystems, cdouble* ySystems) {
    int total = checkedBatchElements(F.n, F.nsystems, "ensemble fused affine action");
    fused_affine_ensemble_spmv_kernel<<<blocksFor(total), THREADS>>>(
        F.n, F.nsystems, F.nnz, F.nctrls, F.ndrifts, driftIndex,
        F.rowPtr, F.colIdx, F.driftValues, F.controlValues, dCoeff,
        xSystems, ySystems);
    cudaCheck(cudaGetLastError(), "fused_affine_ensemble_spmv_kernel");
}

// Applies an on-the-fly affine Liouvillian to a batch of vectors per member
void fusedAffineActionEnsembleBatch(const FusedCsrDevice& F, int driftIndex,
                                    const double* dCoeff, const cdouble* xBatch,
                                    cdouble* yBatch, int nvecPerSystem) {
    int totalVec = checkedBatchElements(F.nsystems, nvecPerSystem, "ensemble vector batch count");
    int total = checkedBatchElements(F.n, totalVec, "ensemble fused affine batched action");
    fused_affine_ensemble_spmv_batch_kernel<<<blocksFor(total), THREADS>>>(
        F.n, F.nsystems, nvecPerSystem, F.nnz, F.nctrls, F.ndrifts, driftIndex,
        F.rowPtr, F.colIdx, F.driftValues, F.controlValues, dCoeff,
        xBatch, yBatch);
    cudaCheck(cudaGetLastError(), "fused_affine_ensemble_spmv_batch_kernel");
}

// Applies a block of control operators to a batch of vectors per member
void fusedControlActionEnsembleBatch(const FusedCsrDevice& F, int firstCtrl,
                                     int nvecPerSystem, const cdouble* xSystems,
                                     cdouble* yBatch) {
    int totalVec = checkedBatchElements(F.nsystems, nvecPerSystem, "ensemble control batch count");
    int total = checkedBatchElements(F.n, totalVec, "ensemble fused control batched action");
    fused_control_ensemble_spmv_batch_kernel<<<blocksFor(total), THREADS>>>(
        F.n, F.nsystems, nvecPerSystem, F.nnz, firstCtrl,
        F.rowPtr, F.colIdx, F.controlValues, xSystems, yBatch);
    cudaCheck(cudaGetLastError(), "fused_control_ensemble_spmv_batch_kernel");
}

// Assembles the affine Liouvillian of every ensemble member once per Taylor action
void assembleAffineEnsemble(const FusedCsrDevice& F, int driftIndex,
                            const double* dCoeff, cdouble* affineValues) {
    if (F.useSell) {
        int total = checkedBatchElements(F.nsystems, F.sellNnzPadded,
                                         "SELL ensemble affine assembly");
        assemble_affine_ensemble_sell_kernel<<<blocksFor(total), THREADS>>>(
            F.nsystems, F.sellNnzPadded, F.nctrls, F.ndrifts, driftIndex,
            F.sellDriftValues, F.sellControlValues, dCoeff, affineValues);
        cudaCheck(cudaGetLastError(), "assemble_affine_ensemble_sell_kernel");
        return;
    }
    int total = checkedBatchElements(F.nsystems, F.nnz, "ensemble affine assembly");
    assemble_affine_ensemble_kernel<<<blocksFor(total), THREADS>>>(
        F.nsystems, F.nnz, F.nctrls, F.ndrifts, driftIndex,
        F.driftValues, F.controlValues, dCoeff, affineValues);
    cudaCheck(cudaGetLastError(), "assemble_affine_ensemble_kernel");
}

// Applies preassembled ensemble affine values to one vector per member
void affineActionEnsemble(const FusedCsrDevice& F, const cdouble* affineValues,
                          const cdouble* xSystems, cdouble* ySystems) {
    int total = checkedBatchElements(F.n, F.nsystems, "preassembled ensemble affine action");
    if (F.useSell) {
        affine_ensemble_spmv_sell_kernel<<<blocksFor(total), THREADS>>>(
            F.n, F.nsystems, F.sellNnzPadded, F.sellSlicePtr,
            F.sellSliceWidth, F.sellColIdx, affineValues, xSystems, ySystems);
        cudaCheck(cudaGetLastError(), "affine_ensemble_spmv_sell_kernel");
        return;
    }
    if (F.useHalfWarpRows) {
        affine_ensemble_spmv_halfwarp_kernel<<<blocksForHalfWarpGroups(total), THREADS>>>(
            F.n, F.nsystems, F.nnz, F.rowPtr, F.colIdx, affineValues,
            xSystems, ySystems);
        cudaCheck(cudaGetLastError(), "affine_ensemble_spmv_halfwarp_kernel");
        return;
    }
    affine_ensemble_spmv_kernel<<<blocksFor(total), THREADS>>>(
        F.n, F.nsystems, F.nnz, F.rowPtr, F.colIdx, affineValues,
        xSystems, ySystems);
    cudaCheck(cudaGetLastError(), "affine_ensemble_spmv_kernel");
}

// Applies preassembled ensemble affine values to a batch of vectors per member
void affineActionEnsembleBatch(const FusedCsrDevice& F, const cdouble* affineValues,
                               const cdouble* xBatch, cdouble* yBatch,
                               int nvecPerSystem) {
    int totalVec = checkedBatchElements(F.nsystems, nvecPerSystem, "preassembled ensemble affine vector count");
    int total = checkedBatchElements(F.n, totalVec, "preassembled ensemble affine batch action");
    if (F.useSell) {
        affine_ensemble_spmv_batch_sell_kernel<<<blocksFor(total), THREADS>>>(
            F.n, F.nsystems, nvecPerSystem, F.sellNnzPadded,
            F.sellSlicePtr, F.sellSliceWidth, F.sellColIdx, affineValues,
            xBatch, yBatch);
        cudaCheck(cudaGetLastError(), "affine_ensemble_spmv_batch_sell_kernel");
        return;
    }
    if (F.useHalfWarpRows) {
        affine_ensemble_spmv_batch_halfwarp_kernel<<<blocksForHalfWarpGroups(total), THREADS>>>(
            F.n, F.nsystems, nvecPerSystem, F.nnz, F.rowPtr, F.colIdx,
            affineValues, xBatch, yBatch);
        cudaCheck(cudaGetLastError(), "affine_ensemble_spmv_batch_halfwarp_kernel");
        return;
    }
    affine_ensemble_spmv_batch_kernel<<<blocksFor(total), THREADS>>>(
        F.n, F.nsystems, nvecPerSystem, F.nnz, F.rowPtr, F.colIdx,
        affineValues, xBatch, yBatch);
    cudaCheck(cudaGetLastError(), "affine_ensemble_spmv_batch_kernel");
}

// Advances the coupled ensemble recurrence using preassembled affine values
void implicitFrechetActionEnsembleBatchAffine(const FusedCsrDevice& F,
                                              const cdouble* affineValues,
                                              int firstCtrl, int nvecPerSystem,
                                              const cdouble* etaBatch,
                                              const cdouble* rhoSystems,
                                              cdouble* outEtaBatch,
                                              cdouble* outRhoSystems) {
    int totalVec = checkedBatchElements(F.nsystems, nvecPerSystem, "preassembled implicit Frechet vector count");
    int total = checkedBatchElements(F.n, totalVec, "preassembled implicit Frechet action");
    if (F.useSell) {
        implicit_frechet_ensemble_affine_batch_sell_kernel<<<blocksFor(total), THREADS>>>(
            F.n, F.nsystems, nvecPerSystem, F.sellNnzPadded, firstCtrl,
            F.sellSlicePtr, F.sellSliceWidth, F.sellColIdx, affineValues,
            F.sellControlValues, etaBatch, rhoSystems, outEtaBatch,
            outRhoSystems);
        cudaCheck(cudaGetLastError(), "implicit_frechet_ensemble_affine_batch_sell_kernel");
        return;
    }
    if (F.useHalfWarpRows) {
        implicit_frechet_ensemble_affine_batch_halfwarp_kernel<<<blocksForHalfWarpGroups(total), THREADS>>>(
            F.n, F.nsystems, nvecPerSystem, F.nnz, firstCtrl,
            F.rowPtr, F.colIdx, affineValues, F.controlValues, etaBatch,
            rhoSystems, outEtaBatch, outRhoSystems);
        cudaCheck(cudaGetLastError(), "implicit_frechet_ensemble_affine_batch_halfwarp_kernel");
        return;
    }
    implicit_frechet_ensemble_affine_batch_kernel<<<blocksFor(total), THREADS>>>(
        F.n, F.nsystems, nvecPerSystem, F.nnz, firstCtrl,
        F.rowPtr, F.colIdx, affineValues, F.controlValues, etaBatch,
        rhoSystems, outEtaBatch, outRhoSystems);
    cudaCheck(cudaGetLastError(), "implicit_frechet_ensemble_affine_batch_kernel");
}

// Advances the coupled ensemble recurrence, assembling the affine values on the fly
void fusedImplicitFrechetActionEnsembleBatch(const FusedCsrDevice& F, int driftIndex,
                                             const double* dCoeff, int firstCtrl,
                                             int nvecPerSystem,
                                             const cdouble* etaBatch,
                                             const cdouble* rhoSystems,
                                             cdouble* outEtaBatch,
                                             cdouble* outRhoSystems) {
    int totalVec = checkedBatchElements(F.nsystems, nvecPerSystem, "ensemble implicit Frechet vector count");
    int total = checkedBatchElements(F.n, totalVec, "ensemble implicit Frechet fused action");
    fused_implicit_frechet_ensemble_batch_kernel<<<blocksFor(total), THREADS>>>(
        F.n, F.nsystems, nvecPerSystem, F.nnz, F.nctrls, F.ndrifts,
        driftIndex, firstCtrl, F.rowPtr, F.colIdx, F.driftValues,
        F.controlValues, dCoeff, etaBatch, rhoSystems, outEtaBatch,
        outRhoSystems);
    cudaCheck(cudaGetLastError(), "fused_implicit_frechet_ensemble_batch_kernel");
}

// Applies the affine Liouvillian, choosing the fused or the per-operator route
void operatorAction(const OperatorPack* pack, int driftIndex, const double* hCoeff,
                    const double* dCoeff, bool adjoint, const cdouble* x, cdouble* y) {
    const FusedCsrDevice& F = adjoint ? pack->fusedAdj : pack->fused;
    if (F.rowPtr != nullptr && F.structurallyComplete && dCoeff != nullptr) {
        fusedAffineAction(F, driftIndex, dCoeff, x, y);
        return;
    }
    fillDevice(y, pack->n);
    const CsrDevice& drift = adjoint ? pack->driftsAdj[driftIndex] : pack->drifts[driftIndex];
    spmvAccum(drift, 1.0, x, y);
    const std::vector<CsrDevice>& ctrls = adjoint ? pack->controlsAdj : pack->controls;
    for (int k = 0; k < pack->nctrls; ++k) {
        if (hCoeff[k] != 0.0) spmvAccum(ctrls[k], hCoeff[k], x, y);
    }
}

// Applies the affine Liouvillian to a batch of vectors, fused where possible
void operatorActionBatch(const OperatorPack* pack, int driftIndex, const double* hCoeff,
                         const double* dCoeff, bool adjoint, const cdouble* xBatch, cdouble* yBatch,
                         int nvec) {
    const FusedCsrDevice& F = adjoint ? pack->fusedAdj : pack->fused;
    if (F.rowPtr != nullptr && F.structurallyComplete && dCoeff != nullptr) {
        fusedAffineActionBatch(F, driftIndex, dCoeff, xBatch, yBatch, nvec);
        return;
    }
    int total = checkedBatchElements(pack->n, nvec, "batched operator action");
    fillDevice(yBatch, total);
    const CsrDevice& drift = adjoint ? pack->driftsAdj[driftIndex] : pack->drifts[driftIndex];
    spmvAccumBatchSameMatrix(drift, 1.0, xBatch, yBatch, nvec);
    const std::vector<CsrDevice>& ctrls = adjoint ? pack->controlsAdj : pack->controls;
    for (int k = 0; k < pack->nctrls; ++k) {
        if (hCoeff[k] != 0.0) spmvAccumBatchSameMatrix(ctrls[k], hCoeff[k], xBatch, yBatch, nvec);
    }
}

// Applies one control operator to a vector
void controlAction(const OperatorPack* pack, int ctrlIndex, bool adjoint,
                   const cdouble* x, cdouble* y) {
    fillDevice(y, pack->n);
    const CsrDevice& C = adjoint ? pack->controlsAdj[ctrlIndex] : pack->controls[ctrlIndex];
    spmvAccum(C, 1.0, x, y);
}

// Applies a block of control operators, one vector of the batch each
void controlActionBatch(const OperatorPack* pack, int firstCtrl, int nvec, bool adjoint,
                        const cdouble* x, cdouble* yBatch) {
    const FusedCsrDevice& F = adjoint ? pack->fusedAdj : pack->fused;
    if (F.rowPtr != nullptr && F.structurallyComplete && F.controlValues != nullptr) {
        fusedControlActionBatch(F, firstCtrl, nvec, x, yBatch);
        return;
    }
    int total = checkedBatchElements(pack->n, nvec, "batched control action");
    fillDevice(yBatch, total);
    const std::vector<CsrDevice>& ctrls = adjoint ? pack->controlsAdj : pack->controls;
    for (int j = 0; j < nvec; ++j) {
        const CsrDevice& C = ctrls[firstCtrl + j];
        spmvAccum(C, 1.0, x, yBatch + static_cast<size_t>(j) * pack->n);
    }
}

// Largest absolute value of a device vector
double deviceMaxAbs(const cdouble* x, int n, double* dReduce, std::vector<double>& hReduce) {
    int nBlocks = blocksFor(n);
    max_abs_reduce_kernel<<<nBlocks, THREADS>>>(x, n, dReduce);
    cudaCheck(cudaGetLastError(), "max_abs_reduce_kernel");
    hReduce.resize(nBlocks);
    cudaCheck(cudaMemcpy(hReduce.data(), dReduce, nBlocks*sizeof(double), cudaMemcpyDeviceToHost),
              "copy max_abs reduction");
    return *std::max_element(hReduce.begin(), hReduce.end());
}

// Largest absolute value across two device vectors
double deviceMaxAbsPair(const cdouble* a, int na, const cdouble* b, int nb,
                        double* dReduce, std::vector<double>& hReduce) {
    int n = std::max(na, nb);
    int nBlocks = blocksFor(n);
    max_abs_pair_reduce_kernel<<<nBlocks, THREADS>>>(a, na, b, nb, dReduce);
    cudaCheck(cudaGetLastError(), "max_abs_pair_reduce_kernel");
    hReduce.resize(nBlocks);
    cudaCheck(cudaMemcpy(hReduce.data(), dReduce, nBlocks*sizeof(double), cudaMemcpyDeviceToHost),
              "copy max_abs_pair reduction");
    return *std::max_element(hReduce.begin(), hReduce.end());
}

// Copies a device vector and returns its largest absolute value in one pass
double copyAndMaxAbs(const cdouble* src, cdouble* dst, int n,
                     double* dReduce, std::vector<double>& hReduce) {
    int nBlocks = blocksFor(n);
    copy_max_abs_reduce_kernel<<<nBlocks, THREADS>>>(src, dst, n, dReduce);
    cudaCheck(cudaGetLastError(), "copy_max_abs_reduce_kernel");
    hReduce.resize(nBlocks);
    cudaCheck(cudaMemcpy(hReduce.data(), dReduce, nBlocks*sizeof(double), cudaMemcpyDeviceToHost),
              "copy copy_max_abs reduction");
    return *std::max_element(hReduce.begin(), hReduce.end());
}

// Copies two device vectors and returns their largest absolute value in one pass
double copyPairAndMaxAbs(const cdouble* srcA, cdouble* dstA, int nA,
                         const cdouble* srcB, cdouble* dstB, int nB,
                         double* dReduce, std::vector<double>& hReduce) {
    int n = std::max(nA, nB);
    int nBlocks = blocksFor(n);
    copy_pair_max_abs_reduce_kernel<<<nBlocks, THREADS>>>(srcA, dstA, nA,
        srcB, dstB, nB, dReduce);
    cudaCheck(cudaGetLastError(), "copy_pair_max_abs_reduce_kernel");
    hReduce.resize(nBlocks);
    cudaCheck(cudaMemcpy(hReduce.data(), dReduce, nBlocks*sizeof(double), cudaMemcpyDeviceToHost),
              "copy copy_pair_max_abs reduction");
    return *std::max_element(hReduce.begin(), hReduce.end());
}

}  // namespace

#endif  // GRAPE_GPU_ACTIONS_CUH
