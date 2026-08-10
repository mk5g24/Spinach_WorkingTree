/*
 * grape_gpu_packs.cuh
 *
 * Construction and teardown of persistent GPU operator packs.
 *
 * MATLAB sparse matrices are converted to CSR, the union sparsity pattern
 * of the drift and control set is built, SELL-32 storage is adopted when
 * its padding ratio is acceptable, and the result is uploaded once and
 * reused by every subsequent objective evaluation.  Adjoint operators are
 * built alongside the forward ones for the backward trajectory.
 */

#ifndef GRAPE_GPU_PACKS_CUH
#define GRAPE_GPU_PACKS_CUH

#include "grape_gpu_propagate.cuh"
#include "grape_gpu_grumble.cuh"

namespace {

// Converts a MEX complex or double to a CUDA complex double
cdouble readSparseValue(const mxArray* A, mwIndex p) {
    if (mxIsComplex(A)) {
        const mxComplexDouble* z = mxGetComplexDoubles(A);
        return make_cd(z[p].real, z[p].imag);
    }
    const double* x = mxGetDoubles(A);
    return make_cd(x[p], 0.0);
}

// Convert a sparse matrix stored in MATLABs internal CCS format to a CSR object
CsrHost sparseToCsr(const mxArray* A, bool adjoint) {

    // Retrieve the indices of the positions of the nnz elements and deduce nnz
    mwSize n = mxGetN(A);
    const mwIndex* jc = mxGetJc(A);
    const mwIndex* ir = mxGetIr(A);
    mwSize nnz = jc[n];

    // Create CsrHost object of dimension nxn and nnz complex zeros
    CsrHost H;
    H.n = static_cast<int>(n);
    H.nnz = static_cast<int>(nnz);
    H.rowPtr.assign(H.n + 1, 0);
    H.colIdx.assign(H.nnz, 0);
    H.values.assign(H.nnz, make_cd(0.0, 0.0));

    // Count row entries in the requested orientation
    for (mwIndex col = 0; col < n; ++col) {
        for (mwIndex p = jc[col]; p < jc[col+1]; ++p) {
            mwIndex row = ir[p];
            int outRow = adjoint ? static_cast<int>(col) : static_cast<int>(row);
            H.rowPtr[outRow + 1]++;
        }
    }
    for (int r = 0; r < H.n; ++r) H.rowPtr[r+1] += H.rowPtr[r];

    // Copy row entries into the CsrHost object in the requested orientation
    std::vector<int> cursor = H.rowPtr;
    std::vector<double> rowAbs(H.n, 0.0);
    for (mwIndex col = 0; col < n; ++col) {
        for (mwIndex p = jc[col]; p < jc[col+1]; ++p) {
            mwIndex row = ir[p];
            cdouble val = readSparseValue(A, p);
            int outRow = adjoint ? static_cast<int>(col) : static_cast<int>(row);
            int outCol = adjoint ? static_cast<int>(row) : static_cast<int>(col);
            if (adjoint) val = conj_cd(val);
            int q = cursor[outRow]++;
            H.colIdx[q] = outCol;
            H.values[q] = val;
            rowAbs[outRow] += abs_cd(val);
        }
    }
    H.infNorm = rowAbs.empty() ? 0.0 : *std::max_element(rowAbs.begin(), rowAbs.end());
    return H;
}

// Upload Csr object to device
CsrDevice uploadCsr(const CsrHost& H) {
    CsrDevice D;
    D.n = H.n;
    D.nnz = H.nnz;
    D.infNorm = H.infNorm;
    cudaCheck(cudaMalloc(&D.rowPtr, (H.n + 1) * sizeof(int)), "cudaMalloc rowPtr");
    cudaCheck(cudaMalloc(&D.colIdx, std::max(1, H.nnz) * sizeof(int)), "cudaMalloc colIdx");
    cudaCheck(cudaMalloc(&D.values, std::max(1, H.nnz) * sizeof(cdouble)), "cudaMalloc values");
    cudaCheck(cudaMemcpy(D.rowPtr, H.rowPtr.data(), (H.n + 1) * sizeof(int), cudaMemcpyHostToDevice),
              "copy rowPtr");
    if (H.nnz > 0) {
        cudaCheck(cudaMemcpy(D.colIdx, H.colIdx.data(), H.nnz * sizeof(int), cudaMemcpyHostToDevice),
                  "copy colIdx");
        cudaCheck(cudaMemcpy(D.values, H.values.data(), H.nnz * sizeof(cdouble), cudaMemcpyHostToDevice),
                  "copy values");
    }
    return D;
}

// Build a Fused CSR object (containing drifts and control operator)
FusedCsrHost buildFusedCsr(const std::vector<CsrHost>& drifts,
                           const std::vector<CsrHost>& controls,
                           int nsystems = 1, int ndriftsPerSystem = -1,
                           bool forcePaddedUnion = false) {
    FusedCsrHost U;
    U.n = drifts.front().n;
    U.nsystems = nsystems;
    U.ndrifts = (ndriftsPerSystem > 0) ? ndriftsPerSystem : static_cast<int>(drifts.size());
    U.nctrls = static_cast<int>(controls.size());

    std::vector<std::map<int, int>> rows(U.n);
    auto addPattern = [&](const CsrHost& H) {
        for (int r = 0; r < H.n; ++r) {
            for (int p = H.rowPtr[r]; p < H.rowPtr[r+1]; ++p) rows[r][H.colIdx[p]] = 0;
        }
    };
    for (const auto& H : drifts) addPattern(H);
    for (const auto& H : controls) addPattern(H);

    U.rowPtr.assign(U.n + 1, 0);
    size_t running = 0;
    for (int r = 0; r < U.n; ++r) {
        running += rows[r].size();
        if (running > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("fused CSR union pattern exceeds 32-bit index range.");
        }
        U.rowPtr[r+1] = static_cast<int>(running);
    }
    U.nnz = U.rowPtr[U.n];
    U.colIdx.assign(U.nnz, 0);
    for (int r = 0; r < U.n; ++r) {
        int q = U.rowPtr[r];
        for (auto& item : rows[r]) {
            item.second = q;
            U.colIdx[q] = item.first;
            ++q;
        }
    }
    for (int r = 0; r < U.n; ++r) {
        U.maxRowNnz = std::max(U.maxRowNnz, U.rowPtr[r+1] - U.rowPtr[r]);
    }
    double avgRowNnz = (U.n > 0) ? static_cast<double>(U.nnz) / static_cast<double>(U.n) : 0.0;
    U.useHalfWarpRows = forcePaddedUnion && U.nsystems > 1 &&
                        avgRowNnz >= static_cast<double>(HALFWARP_ROW_NNZ_THRESHOLD);

    auto matchesUnionPattern = [&](const CsrHost& H) {
        if (H.nnz != U.nnz) return false;
        for (int r = 0; r < U.n; ++r) {
            if ((H.rowPtr[r+1] - H.rowPtr[r]) != (U.rowPtr[r+1] - U.rowPtr[r])) return false;
            for (int p = U.rowPtr[r]; p < U.rowPtr[r+1]; ++p) {
                int hp = H.rowPtr[r] + (p - U.rowPtr[r]);
                if (H.colIdx[hp] != U.colIdx[p]) return false;
            }
        }
        return true;
    };
    U.structurallyComplete = true;
    for (const auto& H : drifts) U.structurallyComplete = U.structurallyComplete && matchesUnionPattern(H);
    for (const auto& H : controls) U.structurallyComplete = U.structurallyComplete && matchesUnionPattern(H);
    if (!U.structurallyComplete && !forcePaddedUnion) return U;
    U.structurallyComplete = true;

    U.driftValues.assign(static_cast<size_t>(U.nsystems) * U.ndrifts * U.nnz, make_cd(0.0, 0.0));
    U.controlValues.assign(static_cast<size_t>(U.nctrls) * U.nnz, make_cd(0.0, 0.0));

    auto scatter = [&](const CsrHost& H, std::vector<cdouble>& values, int opIndex) {
        size_t base = static_cast<size_t>(opIndex) * U.nnz;
        for (int r = 0; r < H.n; ++r) {
            for (int p = H.rowPtr[r]; p < H.rowPtr[r+1]; ++p) {
                auto it = rows[r].find(H.colIdx[p]);
                size_t q = base + static_cast<size_t>(it->second);
                values[q] = add_cd(values[q], H.values[p]);
            }
        }
    };
    for (int i = 0; i < static_cast<int>(drifts.size()); ++i) scatter(drifts[i], U.driftValues, i);
    for (int k = 0; k < U.nctrls; ++k) scatter(controls[k], U.controlValues, k);

    if (forcePaddedUnion && U.nsystems > 1 && U.n >= SELL_SLICE_HEIGHT && U.nnz > 0) {
        U.sellNSlices = (U.n + SELL_SLICE_HEIGHT - 1) / SELL_SLICE_HEIGHT;
        U.sellSlicePtr.assign(U.sellNSlices + 1, 0);
        U.sellSliceWidth.assign(U.sellNSlices, 0);
        size_t padded = 0;
        for (int s = 0; s < U.sellNSlices; ++s) {
            int firstRow = s * SELL_SLICE_HEIGHT;
            int lastRow = std::min(U.n, firstRow + SELL_SLICE_HEIGHT);
            int width = 0;
            for (int r = firstRow; r < lastRow; ++r) {
                width = std::max(width, U.rowPtr[r+1] - U.rowPtr[r]);
            }
            U.sellSliceWidth[s] = width;
            padded += static_cast<size_t>(width) * SELL_SLICE_HEIGHT;
            if (padded > static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("SELL-padded fused CSR pattern exceeds 32-bit index range.");
            }
            U.sellSlicePtr[s+1] = static_cast<int>(padded);
        }
        double sellPaddingRatio = static_cast<double>(padded) / static_cast<double>(U.nnz);
        if (padded > 0 && sellPaddingRatio <= SELL_MAX_PADDING_RATIO) {
            U.useSell = true;
            U.useHalfWarpRows = false;
            U.sellNnzPadded = static_cast<int>(padded);
            U.sellColIdx.assign(U.sellNnzPadded, -1);
            for (int r = 0; r < U.n; ++r) {
                int slice = r / SELL_SLICE_HEIGHT;
                int local = r - slice * SELL_SLICE_HEIGHT;
                for (int p = U.rowPtr[r]; p < U.rowPtr[r+1]; ++p) {
                    int slot = p - U.rowPtr[r];
                    int q = U.sellSlicePtr[slice] + slot * SELL_SLICE_HEIGHT + local;
                    U.sellColIdx[q] = U.colIdx[p];
                }
            }

            U.sellDriftValues.assign(static_cast<size_t>(U.nsystems) * U.ndrifts * U.sellNnzPadded,
                                     make_cd(0.0, 0.0));
            U.sellControlValues.assign(static_cast<size_t>(U.nctrls) * U.sellNnzPadded,
                                       make_cd(0.0, 0.0));
            auto csrToSell = [&](const std::vector<cdouble>& csrValues,
                                 std::vector<cdouble>& sellValues, int nOps) {
                for (int op = 0; op < nOps; ++op) {
                    size_t csrBase = static_cast<size_t>(op) * U.nnz;
                    size_t sellBase = static_cast<size_t>(op) * U.sellNnzPadded;
                    for (int r = 0; r < U.n; ++r) {
                        int slice = r / SELL_SLICE_HEIGHT;
                        int local = r - slice * SELL_SLICE_HEIGHT;
                        for (int p = U.rowPtr[r]; p < U.rowPtr[r+1]; ++p) {
                            int slot = p - U.rowPtr[r];
                            int q = U.sellSlicePtr[slice] + slot * SELL_SLICE_HEIGHT + local;
                            sellValues[sellBase + q] = csrValues[csrBase + p];
                        }
                    }
                }
            };
            csrToSell(U.driftValues, U.sellDriftValues, U.nsystems * U.ndrifts);
            csrToSell(U.controlValues, U.sellControlValues, U.nctrls);
        } else {
            U.sellNSlices = 0;
            U.sellSlicePtr.clear();
            U.sellSliceWidth.clear();
        }
    }
    return U;
}

//Upload fused CSR object to device
FusedCsrDevice uploadFusedCsr(const FusedCsrHost& H) {
    FusedCsrDevice D;
    if (!H.structurallyComplete) return D;
    D.n = H.n;
    D.nnz = H.nnz;
    D.nsystems = H.nsystems;
    D.ndrifts = H.ndrifts;
    D.nctrls = H.nctrls;
    D.maxRowNnz = H.maxRowNnz;
    D.useHalfWarpRows = H.useHalfWarpRows;
    D.useSell = H.useSell;
    D.sellNnzPadded = H.sellNnzPadded;
    D.sellNSlices = H.sellNSlices;
    D.structurallyComplete = H.structurallyComplete;
    cudaCheck(cudaMalloc(&D.rowPtr, (H.n + 1) * sizeof(int)), "cudaMalloc fused rowPtr");
    cudaCheck(cudaMemcpy(D.rowPtr, H.rowPtr.data(), (H.n + 1) * sizeof(int), cudaMemcpyHostToDevice),
              "copy fused rowPtr");
    if (H.nnz > 0) {
        cudaCheck(cudaMalloc(&D.colIdx, H.nnz * sizeof(int)), "cudaMalloc fused colIdx");
        cudaCheck(cudaMemcpy(D.colIdx, H.colIdx.data(), H.nnz * sizeof(int), cudaMemcpyHostToDevice),
                  "copy fused colIdx");
        size_t driftCount = static_cast<size_t>(H.nsystems) * H.ndrifts * H.nnz;
        size_t controlCount = static_cast<size_t>(H.nctrls) * H.nnz;
        if (driftCount > 0) {
            cudaCheck(cudaMalloc(&D.driftValues, driftCount * sizeof(cdouble)), "cudaMalloc fused drift values");
            cudaCheck(cudaMemcpy(D.driftValues, H.driftValues.data(), driftCount * sizeof(cdouble), cudaMemcpyHostToDevice),
                      "copy fused drift values");
        }
        if (controlCount > 0) {
            cudaCheck(cudaMalloc(&D.controlValues, controlCount * sizeof(cdouble)), "cudaMalloc fused control values");
            cudaCheck(cudaMemcpy(D.controlValues, H.controlValues.data(), controlCount * sizeof(cdouble), cudaMemcpyHostToDevice),
                      "copy fused control values");
        }
        if (H.useSell && H.sellNnzPadded > 0) {
            cudaCheck(cudaMalloc(&D.sellSlicePtr, (H.sellNSlices + 1) * sizeof(int)), "cudaMalloc SELL slicePtr");
            cudaCheck(cudaMalloc(&D.sellSliceWidth, H.sellNSlices * sizeof(int)), "cudaMalloc SELL sliceWidth");
            cudaCheck(cudaMalloc(&D.sellColIdx, H.sellNnzPadded * sizeof(int)), "cudaMalloc SELL colIdx");
            cudaCheck(cudaMemcpy(D.sellSlicePtr, H.sellSlicePtr.data(), (H.sellNSlices + 1) * sizeof(int), cudaMemcpyHostToDevice),
                      "copy SELL slicePtr");
            cudaCheck(cudaMemcpy(D.sellSliceWidth, H.sellSliceWidth.data(), H.sellNSlices * sizeof(int), cudaMemcpyHostToDevice),
                      "copy SELL sliceWidth");
            cudaCheck(cudaMemcpy(D.sellColIdx, H.sellColIdx.data(), H.sellNnzPadded * sizeof(int), cudaMemcpyHostToDevice),
                      "copy SELL colIdx");
            size_t sellDriftCount = static_cast<size_t>(H.nsystems) * H.ndrifts * H.sellNnzPadded;
            size_t sellControlCount = static_cast<size_t>(H.nctrls) * H.sellNnzPadded;
            if (sellDriftCount > 0) {
                cudaCheck(cudaMalloc(&D.sellDriftValues, sellDriftCount * sizeof(cdouble)),
                          "cudaMalloc SELL drift values");
                cudaCheck(cudaMemcpy(D.sellDriftValues, H.sellDriftValues.data(),
                                     sellDriftCount * sizeof(cdouble), cudaMemcpyHostToDevice),
                          "copy SELL drift values");
            }
            if (sellControlCount > 0) {
                cudaCheck(cudaMalloc(&D.sellControlValues, sellControlCount * sizeof(cdouble)),
                          "cudaMalloc SELL control values");
                cudaCheck(cudaMemcpy(D.sellControlValues, H.sellControlValues.data(),
                                     sellControlCount * sizeof(cdouble), cudaMemcpyHostToDevice),
                          "copy SELL control values");
            }
        }
    }
    return D;
}

// Free memory allocated by CSR object and destroy
void freeCsr(CsrDevice& D) {
    if (D.rowPtr) cudaFree(D.rowPtr);
    if (D.colIdx) cudaFree(D.colIdx);
    if (D.values) cudaFree(D.values);
    D = CsrDevice{};
}

// Free memory allocated by fused CSR object and destroy
void freeFusedCsr(FusedCsrDevice& D) {
    if (D.rowPtr) cudaFree(D.rowPtr);
    if (D.colIdx) cudaFree(D.colIdx);
    if (D.driftValues) cudaFree(D.driftValues);
    if (D.controlValues) cudaFree(D.controlValues);
    if (D.sellSlicePtr) cudaFree(D.sellSlicePtr);
    if (D.sellSliceWidth) cudaFree(D.sellSliceWidth);
    if (D.sellColIdx) cudaFree(D.sellColIdx);
    if (D.sellDriftValues) cudaFree(D.sellDriftValues);
    if (D.sellControlValues) cudaFree(D.sellControlValues);
    D = FusedCsrDevice{};
}

// Destroy operator pack
void destroyPack(OperatorPack* pack) {
    if (!pack) return;
    for (auto& A : pack->drifts) freeCsr(A);
    for (auto& A : pack->controls) freeCsr(A);
    for (auto& A : pack->driftsAdj) freeCsr(A);
    for (auto& A : pack->controlsAdj) freeCsr(A);
    freeFusedCsr(pack->fused);
    freeFusedCsr(pack->fusedAdj);
    if (pack->d_dt) cudaFree(pack->d_dt);
    delete pack;
}

// Clear all packs
void clearAllPacks() {
    for (auto& item : g_packs) destroyPack(item.second);
    g_packs.clear();
    while (mexIsLocked()) mexUnlock();
}

// When the calcuation is done, clear all remaining packs
void atExitCleanup() {
    clearAllPacks();
}

// Create an operator pack from inputs (drifts, controls and time slices)
OperatorPack* createPackFromInputs(const mxArray* driftsArg, const mxArray* controlsArg, const mxArray* dtArg) {

    // Check consistency
    grumblePackInputs(driftsArg, controlsArg, dtArg);

    auto* pack = new OperatorPack();
    try {
        pack->ndrifts = static_cast<int>(mxGetNumberOfElements(driftsArg));
        pack->nctrls = static_cast<int>(mxGetNumberOfElements(controlsArg));
        pack->nsteps = static_cast<int>(mxGetNumberOfElements(dtArg));

        std::vector<CsrHost> hDrifts;
        std::vector<CsrHost> hControls;
        std::vector<CsrHost> hDriftsAdj;
        std::vector<CsrHost> hControlsAdj;
        hDrifts.reserve(pack->ndrifts);
        hControls.reserve(pack->nctrls);
        hDriftsAdj.reserve(pack->ndrifts);
        hControlsAdj.reserve(pack->nctrls);

        for (int i = 0; i < pack->ndrifts; ++i) {
            const mxArray* A = mxGetCell(driftsArg, i);
            CsrHost H = sparseToCsr(A, false);
            CsrHost HA = sparseToCsr(A, true);
            pack->n = H.n;
            hDrifts.push_back(std::move(H));
            hDriftsAdj.push_back(std::move(HA));
        }

        for (int k = 0; k < pack->nctrls; ++k) {
            const mxArray* C = mxGetCell(controlsArg, k);
            CsrHost H = sparseToCsr(C, false);
            CsrHost HA = sparseToCsr(C, true);
            hControls.push_back(std::move(H));
            hControlsAdj.push_back(std::move(HA));
        }

        for (const auto& H : hDrifts) pack->drifts.push_back(uploadCsr(H));
        for (const auto& H : hControls) pack->controls.push_back(uploadCsr(H));
        for (const auto& H : hDriftsAdj) pack->driftsAdj.push_back(uploadCsr(H));
        for (const auto& H : hControlsAdj) pack->controlsAdj.push_back(uploadCsr(H));

        pack->fused = uploadFusedCsr(buildFusedCsr(hDrifts, hControls));
        pack->fusedAdj = uploadFusedCsr(buildFusedCsr(hDriftsAdj, hControlsAdj));

        cudaCheck(cudaMalloc(&pack->d_dt, pack->nsteps * sizeof(double)), "cudaMalloc dt");
        cudaCheck(cudaMemcpy(pack->d_dt, mxGetDoubles(dtArg), pack->nsteps * sizeof(double), cudaMemcpyHostToDevice),
                  "copy dt");
        return pack;
    } catch (...) {
        destroyPack(pack);
        throw;
    }
}

// Create an ensemble operator pack, one drift set per ensemble member
OperatorPack* createEnsemblePackFromInputs(const mxArray* driftsEnsembleArg,
                                           const mxArray* controlsArg,
                                           const mxArray* dtArg) {

    // Check consistency
    grumbleEnsembleInputs(driftsEnsembleArg, controlsArg, dtArg);

    auto* pack = new OperatorPack();
    try {
        pack->nsystems = static_cast<int>(mxGetNumberOfElements(driftsEnsembleArg));
        pack->nctrls = static_cast<int>(mxGetNumberOfElements(controlsArg));
        pack->nsteps = static_cast<int>(mxGetNumberOfElements(dtArg));

        const mxArray* firstMember = mxGetCell(driftsEnsembleArg, 0);
        pack->ndrifts = mxIsCell(firstMember) ? static_cast<int>(mxGetNumberOfElements(firstMember)) : 1;

        std::vector<CsrHost> hDrifts;
        std::vector<CsrHost> hControls;
        std::vector<CsrHost> hDriftsAdj;
        std::vector<CsrHost> hControlsAdj;
        hDrifts.reserve(static_cast<size_t>(pack->nsystems) * pack->ndrifts);
        hDriftsAdj.reserve(static_cast<size_t>(pack->nsystems) * pack->ndrifts);
        hControls.reserve(pack->nctrls);
        hControlsAdj.reserve(pack->nctrls);

        for (int sys = 0; sys < pack->nsystems; ++sys) {
            const mxArray* member = mxGetCell(driftsEnsembleArg, sys);
            for (int i = 0; i < pack->ndrifts; ++i) {
                const mxArray* A = mxIsCell(member) ? mxGetCell(member, i) : member;
                CsrHost H = sparseToCsr(A, false);
                CsrHost HA = sparseToCsr(A, true);
                pack->n = H.n;
                hDrifts.push_back(std::move(H));
                hDriftsAdj.push_back(std::move(HA));
            }
        }

        for (int k = 0; k < pack->nctrls; ++k) {
            const mxArray* C = mxGetCell(controlsArg, k);
            CsrHost H = sparseToCsr(C, false);
            CsrHost HA = sparseToCsr(C, true);
            hControls.push_back(std::move(H));
            hControlsAdj.push_back(std::move(HA));
        }

        for (const auto& H : hDrifts) pack->drifts.push_back(uploadCsr(H));
        for (const auto& H : hControls) pack->controls.push_back(uploadCsr(H));
        for (const auto& H : hDriftsAdj) pack->driftsAdj.push_back(uploadCsr(H));
        for (const auto& H : hControlsAdj) pack->controlsAdj.push_back(uploadCsr(H));

        pack->fused = uploadFusedCsr(buildFusedCsr(hDrifts, hControls,
            pack->nsystems, pack->ndrifts, true));
        pack->fusedAdj = uploadFusedCsr(buildFusedCsr(hDriftsAdj, hControlsAdj,
            pack->nsystems, pack->ndrifts, true));

        cudaCheck(cudaMalloc(&pack->d_dt, pack->nsteps * sizeof(double)), "cudaMalloc dt");
        cudaCheck(cudaMemcpy(pack->d_dt, mxGetDoubles(dtArg), pack->nsteps * sizeof(double), cudaMemcpyHostToDevice),
                  "copy dt");
        return pack;
    } catch (...) {
        destroyPack(pack);
        throw;
    }
}

// Wrap a pack handle as a MATLAB scalar uint64
mxArray* makeHandle(std::uint64_t handle) {
    mxArray* out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    static_cast<std::uint64_t*>(mxGetData(out))[0] = handle;
    return out;
}

// Report what this backend implements, so the MATLAB side can decide to trust it
mxArray* makeCapabilities() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    bool ready = (err == cudaSuccess && count > 0);

    const char* fields[] = {
        "name", "version", "cudaBackendReady", "supportsRectangleGradient",
        "supportsMidpointGradient", "supportsExactFrechetGradient",
        "supportsScalarExactGradient", "supportsImplicitExactFrechet",
        "supportsUnionPatternCsrFusion",
        "supportsEnsembleResidentPack", "supportsHessian", "supportsPersistentPack", "deviceCount",
        "exactGradientAlgorithm", "sparseStorage", "message"
    };
    mxArray* caps = mxCreateStructMatrix(1, 1, 16, fields);
    mxSetField(caps, 0, "name", mxCreateString("grape_liouv_gpu_cuda_mex"));
    mxSetField(caps, 0, "version", mxCreateDoubleScalar(11.0));
    mxSetField(caps, 0, "cudaBackendReady", mxCreateLogicalScalar(ready));
    mxSetField(caps, 0, "supportsRectangleGradient", mxCreateLogicalScalar(true));
    mxSetField(caps, 0, "supportsMidpointGradient", mxCreateLogicalScalar(true));
    mxSetField(caps, 0, "supportsExactFrechetGradient", mxCreateLogicalScalar(true));
    mxSetField(caps, 0, "supportsScalarExactGradient", mxCreateLogicalScalar(true));
    mxSetField(caps, 0, "supportsImplicitExactFrechet", mxCreateLogicalScalar(true));
    mxSetField(caps, 0, "supportsUnionPatternCsrFusion", mxCreateLogicalScalar(true));
    mxSetField(caps, 0, "supportsEnsembleResidentPack", mxCreateLogicalScalar(true));
    mxSetField(caps, 0, "supportsHessian", mxCreateLogicalScalar(false));
    mxSetField(caps, 0, "supportsPersistentPack", mxCreateLogicalScalar(true));
    mxSetField(caps, 0, "deviceCount", mxCreateDoubleScalar(static_cast<double>(count)));
    mxSetField(caps, 0, "exactGradientAlgorithm", mxCreateString("preassembled_ensemble_affine_batched_exact_frechet_with_fused_implicit_recurrence_fused_taylor_bookkeeping_sell32_and_auto_halfwarp_rows"));
    mxSetField(caps, 0, "sparseStorage", mxCreateString("ensemble_padded_union_pattern_csr_with_guarded_sell32_plus_scalar_regression_csr"));
    if (ready) {
        mxSetField(caps, 0, "message", mxCreateString(
            "CUDA backend ready: persistent union-pattern CSR packs, ensemble-resident padded CSR packs with guarded SELL-32 value storage, preassembled ensemble affine values, automatic half-warp row kernels for dense-row packs, reordered-Taylor step action with fused bookkeeping kernels, MID, split-action exact Frechet, exact_implicit fused Frechet gradients, and ensemble performance estimates."));
    } else {
        mxSetField(caps, 0, "message", mxCreateString(cudaGetErrorString(err)));
    }
    return caps;
}

// Read a state vector from MATLAB into a dense complex host vector
std::vector<cdouble> readDenseVector(const mxArray* arg, int expectedN, const char* name) {

    // Check consistency
    grumbleStateVector(arg, name, expectedN);

    std::vector<cdouble> out(expectedN);
    if (mxIsSparse(arg)) {
        const mwIndex* jc = mxGetJc(arg);
        const mwIndex* ir = mxGetIr(arg);
        mwSize ncols = mxGetN(arg);
        mwSize nrows = mxGetM(arg);
        for (mwIndex col = 0; col < ncols; ++col) {
            for (mwIndex p = jc[col]; p < jc[col+1]; ++p) {
                mwIndex idx = ir[p] + col * nrows;
                if (idx >= static_cast<mwIndex>(expectedN)) {
                    throw std::runtime_error(std::string(name) + " sparse vector indexing exceeds expected dimension.");
                }
                out[static_cast<size_t>(idx)] = readSparseValue(arg, p);
            }
        }
        return out;
    }
    if (mxIsComplex(arg)) {
        const mxComplexDouble* z = mxGetComplexDoubles(arg);
        for (int i = 0; i < expectedN; ++i) out[i] = make_cd(z[i].real, z[i].imag);
    } else {
        const double* x = mxGetDoubles(arg);
        for (int i = 0; i < expectedN; ++i) out[i] = make_cd(x[i], 0.0);
    }
    return out;
}

// Read a common or per-member state vector into one dense block for the ensemble
std::vector<cdouble> readEnsembleVectors(const mxArray* arg, int n, int nsystems, const char* name) {

    // Check consistency
    grumbleEnsembleVectors(arg, name, n, nsystems);

    mwSize elems = mxGetNumberOfElements(arg);
    if (elems == static_cast<mwSize>(n)) {
        std::vector<cdouble> one = readDenseVector(arg, n, name);
        std::vector<cdouble> out(static_cast<size_t>(n) * nsystems);
        for (int sys = 0; sys < nsystems; ++sys) {
            std::copy(one.begin(), one.end(), out.begin() + static_cast<size_t>(sys) * n);
        }
        return out;
    }
    std::vector<cdouble> out(static_cast<size_t>(n) * nsystems);
    if (mxIsSparse(arg)) {
        const mwIndex* jc = mxGetJc(arg);
        const mwIndex* ir = mxGetIr(arg);
        mwSize ncols = mxGetN(arg);
        for (mwIndex col = 0; col < ncols; ++col) {
            for (mwIndex p = jc[col]; p < jc[col+1]; ++p) {
                out[static_cast<size_t>(col) * n + ir[p]] = readSparseValue(arg, p);
            }
        }
        return out;
    }
    if (mxIsComplex(arg)) {
        const mxComplexDouble* z = mxGetComplexDoubles(arg);
        for (int sys = 0; sys < nsystems; ++sys) {
            for (int i = 0; i < n; ++i) {
                size_t idx = static_cast<size_t>(sys) * n + i;
                out[idx] = make_cd(z[idx].real, z[idx].imag);
            }
        }
    } else {
        const double* x = mxGetDoubles(arg);
        for (int sys = 0; sys < nsystems; ++sys) {
            for (int i = 0; i < n; ++i) {
                size_t idx = static_cast<size_t>(sys) * n + i;
                out[idx] = make_cd(x[idx], 0.0);
            }
        }
    }
    return out;
}

}  // namespace

#endif  // GRAPE_GPU_PACKS_CUH
