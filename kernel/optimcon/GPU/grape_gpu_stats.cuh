/*
 * grape_gpu_stats.cuh
 *
 * Work accounting.  The backend counts the floating point operations, the
 * bytes moved, and the sparse elements visited by the kernels it launches,
 * and hands the totals back to MATLAB so that a benchmark can quote
 * throughput without a hardware counter.
 *
 * These are estimates of the dominant sparse work: complex sparse actions,
 * affine assembly, and inner products.  They deliberately exclude launch
 * overhead and anything happening on the MATLAB side, so they answer the
 * question of whether a case is compute-bound or memory-bound rather than
 * measuring the machine.
 */

#ifndef GRAPE_GPU_STATS_CUH
#define GRAPE_GPU_STATS_CUH

#include "grape_gpu_types.cuh"

namespace {

// Counts the work done by one affine assembly pass
void addAffineAssemblyStats(EvalStats* stats, const FusedCsrDevice& F) {
    if (!stats || F.nnz <= 0 || F.nsystems <= 0) return;
    double values = static_cast<double>(F.nnz) * F.nsystems;
    stats->affineAssemblyCalls += 1.0;
    stats->affineAssemblyValues += values;
    stats->flops += values * F.nctrls * 4.0;
    stats->bytes += values * (sizeof(cdouble) + sizeof(cdouble)) +
                    values * F.nctrls * (sizeof(cdouble) + sizeof(double));
}

// Counts the work done by one sparse affine action
void addAffineSpmvStats(EvalStats* stats, const FusedCsrDevice& F, int nvecPerSystem) {
    if (!stats || F.nnz <= 0 || F.nsystems <= 0 || nvecPerSystem <= 0) return;
    double nnzVisits = static_cast<double>(F.nnz) * F.nsystems * nvecPerSystem;
    double rows = static_cast<double>(F.n) * F.nsystems * nvecPerSystem;
    stats->sparseKernelCalls += 1.0;
    stats->sparseNnzVisits += nnzVisits;
    stats->flops += nnzVisits * 8.0;
    stats->bytes += nnzVisits * (2.0*sizeof(cdouble) + sizeof(int)) + rows * sizeof(cdouble);
}

// Counts the work done by one fused implicit Frechet action
void addImplicitFrechetStats(EvalStats* stats, const FusedCsrDevice& F, int nvecPerSystem) {
    if (!stats || F.nnz <= 0 || F.nsystems <= 0 || nvecPerSystem <= 0) return;
    double base = static_cast<double>(F.nnz) * F.nsystems;
    double rows = static_cast<double>(F.n) * F.nsystems;
    stats->sparseKernelCalls += 1.0;
    stats->sparseNnzVisits += base * (2.0*nvecPerSystem + 1.0);
    stats->flops += base * (16.0*nvecPerSystem + 8.0);
    stats->bytes += base * (nvecPerSystem * (3.0*sizeof(cdouble) + sizeof(int)) +
                            (2.0*sizeof(cdouble) + sizeof(int))) +
                    rows * (nvecPerSystem + 1.0) * sizeof(cdouble);
}

// Counts the work done by one batch of inner products
void addDotStats(EvalStats* stats, int n, int nsystems, int nvecPerSystem) {
    if (!stats || n <= 0 || nsystems <= 0 || nvecPerSystem <= 0) return;
    double elems = static_cast<double>(n) * nsystems * nvecPerSystem;
    stats->reductionKernelCalls += 1.0;
    stats->flops += elems * 8.0;
    stats->bytes += elems * 2.0*sizeof(cdouble) + static_cast<double>(nsystems) * nvecPerSystem * sizeof(cdouble);
}

// Converts the accumulated work counters into a MATLAB structure
mxArray* makePerformanceStruct(const EvalStats& stats) {
    const char* fields[] = {
        "estimated_flops", "estimated_bytes", "estimated_sparse_nnz_visits",
        "affine_assembly_values", "affine_assembly_calls",
        "sparse_kernel_calls", "reduction_kernel_calls"
    };
    mxArray* perf = mxCreateStructMatrix(1, 1, 7, fields);
    mxSetField(perf, 0, "estimated_flops", mxCreateDoubleScalar(stats.flops));
    mxSetField(perf, 0, "estimated_bytes", mxCreateDoubleScalar(stats.bytes));
    mxSetField(perf, 0, "estimated_sparse_nnz_visits", mxCreateDoubleScalar(stats.sparseNnzVisits));
    mxSetField(perf, 0, "affine_assembly_values", mxCreateDoubleScalar(stats.affineAssemblyValues));
    mxSetField(perf, 0, "affine_assembly_calls", mxCreateDoubleScalar(stats.affineAssemblyCalls));
    mxSetField(perf, 0, "sparse_kernel_calls", mxCreateDoubleScalar(stats.sparseKernelCalls));
    mxSetField(perf, 0, "reduction_kernel_calls", mxCreateDoubleScalar(stats.reductionKernelCalls));
    return perf;
}

}  // namespace

#endif  // GRAPE_GPU_STATS_CUH
