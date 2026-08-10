/*
 * grape_gpu_types.cuh
 *
 * Storage layouts, compile-time constants, mode name lookups, and complex
 * arithmetic helpers shared by every other unit of the CUDA backend.
 *
 * CsrHost/CsrDevice hold one sparse operator. FusedCsrHost/FusedCsrDevice
 * hold the union sparsity pattern of the whole drift and control set, with
 * operator-major value blocks so that an affine Liouvillian can be assembled
 * on the device without touching the index arrays. OperatorPack is the
 * persistent GPU residency unit that survives across optimiser iterations.
 *
 * All units of this backend open the same anonymous namespace, so every
 * symbol below has internal linkage and the whole backend remains a single
 * translation unit with unchanged inlining behaviour.
 */

#ifndef GRAPE_GPU_TYPES_CUH
#define GRAPE_GPU_TYPES_CUH

#include "mex.h"
#include "matrix.h"
#include <cuda_runtime.h>
#include <cuComplex.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using cdouble = cuDoubleComplex;

constexpr int THREADS = 256;
constexpr int SELL_SLICE_HEIGHT = 32;
constexpr int HALFWARP_ROW_NNZ_THRESHOLD = 32;
constexpr double SELL_MAX_PADDING_RATIO = 1.35;
constexpr int MAX_TAYLOR_ITER = 256;
constexpr double TAYLOR_TOL = 2.2204460492503131e-16; // eps('double')

// CSR structure object for host side
struct CsrHost {
    int n = 0;
    int nnz = 0;
    std::vector<int> rowPtr;
    std::vector<int> colIdx;
    std::vector<cdouble> values;
    double infNorm = 0.0;
};

// CSR structure object for device side
struct CsrDevice {
    int n = 0;
    int nnz = 0;
    int* rowPtr = nullptr;
    int* colIdx = nullptr;
    cdouble* values = nullptr;
    double infNorm = 0.0;
};

// Fused CSR object for host side
struct FusedCsrHost {
    int n = 0;
    int nnz = 0;
    int nsystems = 1;
    int ndrifts = 0;
    int nctrls = 0;
    int maxRowNnz = 0;
    bool useHalfWarpRows = false;
    bool useSell = false;
    int sellNnzPadded = 0;
    int sellNSlices = 0;
    bool structurallyComplete = false;
    std::vector<int> rowPtr;
    std::vector<int> colIdx;
    std::vector<cdouble> driftValues;   // operator-major: drift index * nnz + pattern index
    std::vector<cdouble> controlValues; // operator-major: control index * nnz + pattern index
    std::vector<int> sellSlicePtr;       // SELL-C-sigma-like slice offsets, C=SELL_SLICE_HEIGHT
    std::vector<int> sellSliceWidth;
    std::vector<int> sellColIdx;         // slice-major: slicePtr[s] + slot*C + localRow
    std::vector<cdouble> sellDriftValues;
    std::vector<cdouble> sellControlValues;
};

// Fused CSR object for device side
struct FusedCsrDevice {
    int n = 0;
    int nnz = 0;
    int nsystems = 1;
    int ndrifts = 0;
    int nctrls = 0;
    int maxRowNnz = 0;
    bool useHalfWarpRows = false;
    bool useSell = false;
    int sellNnzPadded = 0;
    int sellNSlices = 0;
    bool structurallyComplete = false;
    int* rowPtr = nullptr;
    int* colIdx = nullptr;
    cdouble* driftValues = nullptr;
    cdouble* controlValues = nullptr;
    int* sellSlicePtr = nullptr;
    int* sellSliceWidth = nullptr;
    int* sellColIdx = nullptr;
    cdouble* sellDriftValues = nullptr;
    cdouble* sellControlValues = nullptr;
};

// Operator pack structures
struct OperatorPack {
    int n = 0;
    int nsystems = 1;
    int ndrifts = 0;
    int nctrls = 0;
    int nsteps = 0;
    std::vector<CsrDevice> drifts;
    std::vector<CsrDevice> controls;
    std::vector<CsrDevice> driftsAdj;
    std::vector<CsrDevice> controlsAdj;
    FusedCsrDevice fused;
    FusedCsrDevice fusedAdj;
    double* d_dt = nullptr;
};

// Statistics for evaluation process (number of subfunction calls)
struct EvalStats {
    double flops = 0.0;
    double bytes = 0.0;
    double sparseNnzVisits = 0.0;
    double affineAssemblyValues = 0.0;
    double affineAssemblyCalls = 0.0;
    double sparseKernelCalls = 0.0;
    double reductionKernelCalls = 0.0;
};

// Create g_packs map of operator pack pointer and pack handle
std::map<std::uint64_t, OperatorPack*> g_packs;
std::uint64_t g_nextHandle = 1;

// -------------------------------------------------------------------------
// Small helpers

// Report fail ID and Msg
void fail(const char* id, const std::string& msg) {
    mexErrMsgIdAndTxt(id, "%s", msg.c_str());
}

// CUDA error reporter
void cudaCheck(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(err));
    }
}

// -------------------------------------------------------------------------
// Translate the fidelity type name into the code used by the gradient kernels
int fidelityMode(const std::string& s) {
    if (s == "real") return 0;
    if (s == "imag") return 1;
    if (s == "square") return 2;
    throw std::runtime_error("fidelity_type must be 'real', 'imag', or 'square'.");
}

// True for the cheap midpoint gradient
bool isMidpointGradientMode(const std::string& s) {
    return s == "midpoint" || s == "mid";
}

// True for the unbatched exact Frechet gradient
bool isScalarExactGradientMode(const std::string& s) {
    return s == "exact_scalar" || s == "frechet_scalar";
}

// True for the fused implicit exact Frechet gradient
bool isImplicitExactGradientMode(const std::string& s) {
    return s == "exact_implicit" || s == "frechet_implicit";
}

// True for any of the exact Frechet gradient variants
bool isExactGradientMode(const std::string& s) {
    return s == "exact" || s == "frechet" ||
           isScalarExactGradientMode(s) || isImplicitExactGradientMode(s);
}

// -------------------------------------------------------------------------
// Complex arithmetic helpers, inlined because they sit in the innermost loops

// Complex number built from its real and imaginary parts
__host__ __device__ inline cdouble make_cd(double re, double im) {
    return make_cuDoubleComplex(re, im);
}

// Sum of two complex numbers
__host__ __device__ inline cdouble add_cd(cdouble a, cdouble b) {
    return make_cd(a.x + b.x, a.y + b.y);
}

// Product of two complex numbers
__host__ __device__ inline cdouble mul_cd(cdouble a, cdouble b) {
    return make_cd(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

// Complex number scaled by a real number
__host__ __device__ inline cdouble scale_cd(cdouble a, double s) {
    return make_cd(a.x*s, a.y*s);
}

// Complex conjugate of a complex number
__host__ __device__ inline cdouble conj_cd(cdouble a) {
    return make_cd(a.x, -a.y);
}

// Absolute value of a complex number
__host__ __device__ inline double abs_cd(cdouble a) {
    return hypot(a.x, a.y);
}

// Sum of a complex number across a half-warp
__device__ inline cdouble halfWarpSum(cdouble v, unsigned mask) {
    for (int offset = 8; offset > 0; offset >>= 1) {
        v.x += __shfl_down_sync(mask, v.x, offset, 16);
        v.y += __shfl_down_sync(mask, v.y, offset, 16);
    }
    return v;
}

}

#endif
