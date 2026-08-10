/*
 * grape_gpu_eval.cuh
 *
 * Objective evaluation: fidelity, gradient, and optional forward trajectory.
 *
 * evaluatePack handles a single spin system, evaluateEnsemblePack handles an
 * ensemble-resident pack and returns the ensemble-averaged fidelity and
 * gradient.  Both dispatch on the gradient mode: midpoint, split-action
 * exact Frechet, or fused implicit exact Frechet.  directEvaluate builds a
 * throwaway pack for callers that do not keep a persistent handle.
 */

#ifndef GRAPE_GPU_EVAL_CUH
#define GRAPE_GPU_EVAL_CUH

#include "grape_gpu_packs.cuh"

namespace {

// Copy a complex host vector into a MATLAB complex matrix
mxArray* makeComplexMatrix(int m, int n, const std::vector<cdouble>& h) {
    mxArray* out = mxCreateDoubleMatrix(m, n, mxCOMPLEX);
    mxComplexDouble* z = mxGetComplexDoubles(out);
    for (int i = 0; i < m*n; ++i) {
        z[i].real = h[i].x;
        z[i].imag = h[i].y;
    }
    return out;
}

// Fidelity and gradient of an ensemble pack, averaged over its members
void evaluateEnsemblePack(OperatorPack* pack, const mxArray* waveformArg,
                          const mxArray* rhoInitArg, const mxArray* rhoTargArg,
                          const std::string& fidelityType, const std::string& gradientMode,
                          bool returnForward, int nlhs, mxArray* plhs[]) {

    // Check consistency
    grumbleWaveform(waveformArg, pack);
    grumbleEnsembleRequest(gradientMode, returnForward);

    const int n = pack->n;
    const int nsystems = pack->nsystems;
    const int nsteps = pack->nsteps;
    const int nctrls = pack->nctrls;
    const int stateElems = checkedBatchElements(n, nsystems, "ensemble state storage");
    const int ngrad = checkedBatchElements(nctrls, nsteps, "ensemble gradient size");
    const double* hWave = mxGetDoubles(waveformArg);
    const bool useImplicitFused = isImplicitExactGradientMode(gradientMode);
    std::vector<double> hDt(nsteps);
    cudaCheck(cudaMemcpy(hDt.data(), pack->d_dt, nsteps*sizeof(double), cudaMemcpyDeviceToHost), "copy dt to host");

    std::vector<cdouble> hRhoInit = readEnsembleVectors(rhoInitArg, n, nsystems, "rho_init");
    std::vector<cdouble> hRhoTarg = readEnsembleVectors(rhoTargArg, n, nsystems, "rho_targ");

    cdouble *dFwd=nullptr, *dBwd=nullptr, *dTerm=nullptr, *dWork=nullptr;
    cdouble *dTermRho=nullptr, *dSumRho=nullptr, *dWorkRho=nullptr;
    cdouble *dEtaBatch=nullptr, *dTermEtaBatch=nullptr, *dWorkEtaBatch=nullptr, *dTmpBatch=nullptr;
    cdouble *dOverlap=nullptr, *dGradEns=nullptr, *dAffineValues=nullptr;
    double *dWave=nullptr, *dGradMean=nullptr, *dReduce=nullptr;
    std::vector<double> hReduce;
    EvalStats stats;
    int exactBatchSize = std::max(1, nctrls);

    try {
        cudaCheck(cudaMalloc(&dFwd, static_cast<size_t>(stateElems)*(nsteps+1)*sizeof(cdouble)), "cudaMalloc ensemble fwd");
        cudaCheck(cudaMalloc(&dBwd, static_cast<size_t>(stateElems)*(nsteps+1)*sizeof(cdouble)), "cudaMalloc ensemble bwd");
        cudaCheck(cudaMalloc(&dTerm, stateElems*sizeof(cdouble)), "cudaMalloc ensemble term");
        cudaCheck(cudaMalloc(&dWork, stateElems*sizeof(cdouble)), "cudaMalloc ensemble work");
        cudaCheck(cudaMalloc(&dTermRho, stateElems*sizeof(cdouble)), "cudaMalloc ensemble term rho");
        cudaCheck(cudaMalloc(&dSumRho, stateElems*sizeof(cdouble)), "cudaMalloc ensemble sum rho");
        cudaCheck(cudaMalloc(&dWorkRho, stateElems*sizeof(cdouble)), "cudaMalloc ensemble work rho");
        cudaCheck(cudaMalloc(&dOverlap, nsystems*sizeof(cdouble)), "cudaMalloc ensemble overlap");
        cudaCheck(cudaMalloc(&dGradEns, static_cast<size_t>(ngrad)*nsystems*sizeof(cdouble)), "cudaMalloc ensemble gradients");
        cudaCheck(cudaMalloc(&dGradMean, ngrad*sizeof(double)), "cudaMalloc mean gradient");
        int affineNnz = std::max(pack->fused.useSell ? pack->fused.sellNnzPadded : pack->fused.nnz,
                                 pack->fusedAdj.useSell ? pack->fusedAdj.sellNnzPadded : pack->fusedAdj.nnz);
        if (affineNnz > 0) {
            int affineElems = checkedBatchElements(nsystems, affineNnz, "ensemble affine value buffer");
            cudaCheck(cudaMalloc(&dAffineValues, static_cast<size_t>(affineElems)*sizeof(cdouble)),
                      "cudaMalloc ensemble affine values");
        }

        size_t waveElems = static_cast<size_t>(nctrls) * nsteps;
        if (waveElems > 0) {
            cudaCheck(cudaMalloc(&dWave, waveElems * sizeof(double)), "cudaMalloc waveform");
            cudaCheck(cudaMemcpy(dWave, hWave, waveElems * sizeof(double), cudaMemcpyHostToDevice),
                      "copy waveform");
        }

        auto freeBatchBuffers = [&]() {
            if (dEtaBatch) { cudaFree(dEtaBatch); dEtaBatch = nullptr; }
            if (dTermEtaBatch) { cudaFree(dTermEtaBatch); dTermEtaBatch = nullptr; }
            if (dWorkEtaBatch) { cudaFree(dWorkEtaBatch); dWorkEtaBatch = nullptr; }
            if (dTmpBatch) { cudaFree(dTmpBatch); dTmpBatch = nullptr; }
        };
        auto allocateBatchBuffers = [&](int batchSize) -> cudaError_t {
            freeBatchBuffers();
            int elems = checkedBatchElements(stateElems, batchSize, "ensemble exact-gradient batch allocation");
            size_t bytes = static_cast<size_t>(elems) * sizeof(cdouble);
            cudaError_t err = cudaMalloc(&dEtaBatch, bytes);
            if (err != cudaSuccess) { freeBatchBuffers(); return err; }
            err = cudaMalloc(&dTermEtaBatch, bytes);
            if (err != cudaSuccess) { freeBatchBuffers(); return err; }
            err = cudaMalloc(&dWorkEtaBatch, bytes);
            if (err != cudaSuccess) { freeBatchBuffers(); return err; }
            err = cudaMalloc(&dTmpBatch, bytes);
            if (err != cudaSuccess) { freeBatchBuffers(); return err; }
            return cudaSuccess;
        };
        while (true) {
            cudaError_t err = allocateBatchBuffers(exactBatchSize);
            if (err == cudaSuccess) break;
            if (exactBatchSize == 1) cudaCheck(err, "cudaMalloc ensemble exact-gradient batch buffers");
            exactBatchSize = std::max(1, exactBatchSize / 2);
        }
        int reduceLen = std::max(stateElems, checkedBatchElements(stateElems, exactBatchSize, "ensemble reduction"));
        cudaCheck(cudaMalloc(&dReduce, blocksFor(reduceLen)*sizeof(double)), "cudaMalloc ensemble reduce");

        cudaCheck(cudaMemcpy(dFwd, hRhoInit.data(), stateElems*sizeof(cdouble), cudaMemcpyHostToDevice), "copy ensemble rho_init");
        cudaCheck(cudaMemcpy(dBwd + static_cast<size_t>(stateElems)*nsteps, hRhoTarg.data(), stateElems*sizeof(cdouble), cudaMemcpyHostToDevice), "copy ensemble rho_targ");

        for (int s = 0; s < nsteps; ++s) {
            int driftIndex = s % pack->ndrifts;
            const double* coeff = hWave + static_cast<size_t>(s)*nctrls;
            const double* dCoeff = dWave ? dWave + static_cast<size_t>(s)*nctrls : nullptr;
            propagateStateEnsemble(pack, driftIndex, coeff, dCoeff, false,
                                   dFwd + static_cast<size_t>(stateElems)*s,
                                   dFwd + static_cast<size_t>(stateElems)*(s+1),
                                   hDt[s], dTerm, dWork, dReduce, hReduce,
                                   dAffineValues, &stats);
        }

        for (int s = nsteps-1; s >= 0; --s) {
            int driftIndex = s % pack->ndrifts;
            const double* coeff = hWave + static_cast<size_t>(s)*nctrls;
            const double* dCoeff = dWave ? dWave + static_cast<size_t>(s)*nctrls : nullptr;
            propagateStateEnsemble(pack, driftIndex, coeff, dCoeff, true,
                                   dBwd + static_cast<size_t>(stateElems)*(s+1),
                                   dBwd + static_cast<size_t>(stateElems)*s,
                                   -hDt[s], dTerm, dWork, dReduce, hReduce,
                                   dAffineValues, &stats);
        }

        dot_ensemble_store_kernel<<<nsystems, THREADS>>>(
            dBwd + static_cast<size_t>(stateElems)*nsteps,
            dFwd + static_cast<size_t>(stateElems)*nsteps,
            n, nsystems, dOverlap);
        cudaCheck(cudaGetLastError(), "dot ensemble overlap");
        addDotStats(&stats, n, nsystems, 1);
        std::vector<cdouble> hOverlap(nsystems);
        cudaCheck(cudaMemcpy(hOverlap.data(), dOverlap, nsystems*sizeof(cdouble), cudaMemcpyDeviceToHost), "copy ensemble overlaps");

        bool wantGrad = (nlhs >= 3);
        if (wantGrad) {
            for (int s = 0; s < nsteps; ++s) {
                int driftIndex = s % pack->ndrifts;
                const double* coeff = hWave + static_cast<size_t>(s)*nctrls;
                const double* dCoeff = dWave ? dWave + static_cast<size_t>(s)*nctrls : nullptr;
                for (int firstCtrl = 0; firstCtrl < nctrls; firstCtrl += exactBatchSize) {
                    int nBatch = std::min(exactBatchSize, nctrls - firstCtrl);
                    propagateFrechetEnsembleBatch(pack, driftIndex, coeff, dCoeff, firstCtrl, nBatch,
                                                  dFwd + static_cast<size_t>(stateElems)*s,
                                                  dEtaBatch, hDt[s], dTermEtaBatch,
                                                  dTermRho, dSumRho, dWorkEtaBatch,
                                                  dWorkRho, dTmpBatch, dReduce, hReduce,
                                                  useImplicitFused, dAffineValues, &stats);
                    dot_ensemble_batch_store_kernel<<<nsystems*nBatch, THREADS>>>(
                        dBwd + static_cast<size_t>(stateElems)*(s+1),
                        dEtaBatch, n, nsystems, nBatch, dGradEns, firstCtrl + s*nctrls);
                    cudaCheck(cudaGetLastError(), "dot ensemble batch gradient");
                    addDotStats(&stats, n, nsystems, nBatch);
                }
            }
            int mode = fidelityMode(fidelityType);
            int totalGrad = checkedBatchElements(ngrad, nsystems, "ensemble postprocess gradient");
            postprocess_ensemble_gradient_kernel<<<blocksFor(totalGrad), THREADS>>>(dGradEns, ngrad, nsystems, dOverlap, mode);
            cudaCheck(cudaGetLastError(), "postprocess_ensemble_gradient_kernel");
            average_real_gradient_kernel<<<ngrad, THREADS>>>(dGradEns, dGradMean, ngrad, nsystems);
            cudaCheck(cudaGetLastError(), "average_real_gradient_kernel");
        }

        int mode = fidelityMode(fidelityType);
        double fidReal = 0.0;
        for (const auto& ov : hOverlap) {
            if (mode == 0) fidReal += ov.x;
            else if (mode == 1) fidReal += ov.y;
            else fidReal += ov.x*ov.x + ov.y*ov.y;
        }
        fidReal /= static_cast<double>(nsystems);

        const char* trajFields[] = {"forward", "performance"};
        mxArray* traj = mxCreateStructMatrix(1, 1, 2, trajFields);
        mxSetField(traj, 0, "forward", mxCreateDoubleMatrix(0, 0, mxREAL));
        mxSetField(traj, 0, "performance", makePerformanceStruct(stats));
        if (nlhs >= 1) plhs[0] = traj; else mxDestroyArray(traj);
        if (nlhs >= 2) plhs[1] = mxCreateDoubleScalar(fidReal);
        if (nlhs >= 3) {
            std::vector<double> hGrad(ngrad);
            cudaCheck(cudaMemcpy(hGrad.data(), dGradMean, hGrad.size()*sizeof(double), cudaMemcpyDeviceToHost), "copy mean ensemble gradient");
            mxArray* grad = mxCreateDoubleMatrix(nctrls, nsteps, mxREAL);
            double* g = mxGetDoubles(grad);
            std::copy(hGrad.begin(), hGrad.end(), g);
            plhs[2] = grad;
        }
    } catch (...) {
        if (dFwd) cudaFree(dFwd); if (dBwd) cudaFree(dBwd);
        if (dTerm) cudaFree(dTerm); if (dWork) cudaFree(dWork);
        if (dTermRho) cudaFree(dTermRho); if (dSumRho) cudaFree(dSumRho); if (dWorkRho) cudaFree(dWorkRho);
        if (dEtaBatch) cudaFree(dEtaBatch); if (dTermEtaBatch) cudaFree(dTermEtaBatch);
        if (dWorkEtaBatch) cudaFree(dWorkEtaBatch); if (dTmpBatch) cudaFree(dTmpBatch);
        if (dOverlap) cudaFree(dOverlap); if (dGradEns) cudaFree(dGradEns); if (dGradMean) cudaFree(dGradMean);
        if (dAffineValues) cudaFree(dAffineValues);
        if (dWave) cudaFree(dWave); if (dReduce) cudaFree(dReduce);
        throw;
    }

    if (dFwd) cudaFree(dFwd); if (dBwd) cudaFree(dBwd);
    if (dTerm) cudaFree(dTerm); if (dWork) cudaFree(dWork);
    if (dTermRho) cudaFree(dTermRho); if (dSumRho) cudaFree(dSumRho); if (dWorkRho) cudaFree(dWorkRho);
    if (dEtaBatch) cudaFree(dEtaBatch); if (dTermEtaBatch) cudaFree(dTermEtaBatch);
    if (dWorkEtaBatch) cudaFree(dWorkEtaBatch); if (dTmpBatch) cudaFree(dTmpBatch);
    if (dOverlap) cudaFree(dOverlap); if (dGradEns) cudaFree(dGradEns); if (dGradMean) cudaFree(dGradMean);
    if (dAffineValues) cudaFree(dAffineValues);
    if (dWave) cudaFree(dWave); if (dReduce) cudaFree(dReduce);
}

// Fidelity and gradient of a single-system pack
void evaluatePack(OperatorPack* pack, const mxArray* waveformArg,
                  const mxArray* rhoInitArg, const mxArray* rhoTargArg,
                  const std::string& fidelityType, const std::string& gradientMode,
                  bool returnForward, int nlhs, mxArray* plhs[]) {
    if (pack->nsystems > 1) {
        evaluateEnsemblePack(pack, waveformArg, rhoInitArg, rhoTargArg,
                             fidelityType, gradientMode, returnForward, nlhs, plhs);
        return;
    }

    // Check consistency
    grumbleWaveform(waveformArg, pack);

    const int n = pack->n;
    const int nsteps = pack->nsteps;
    const int nctrls = pack->nctrls;
    const double* hWave = mxGetDoubles(waveformArg);
    const bool useImplicitFused = isImplicitExactGradientMode(gradientMode);
    std::vector<double> hDt(nsteps);
    cudaCheck(cudaMemcpy(hDt.data(), pack->d_dt, nsteps*sizeof(double), cudaMemcpyDeviceToHost), "copy dt to host");

    std::vector<cdouble> hRhoInit = readDenseVector(rhoInitArg, n, "rho_init");
    std::vector<cdouble> hRhoTarg = readDenseVector(rhoTargArg, n, "rho_targ");

    cdouble *dFwd=nullptr, *dBwd=nullptr, *dTerm=nullptr, *dWork=nullptr;
    cdouble *dEta=nullptr, *dTermEta=nullptr, *dTermRho=nullptr, *dSumRho=nullptr;
    cdouble *dWorkEta=nullptr, *dWorkRho=nullptr, *dTmp=nullptr, *dGrad=nullptr;
    cdouble *dEtaBatch=nullptr, *dTermEtaBatch=nullptr, *dWorkEtaBatch=nullptr, *dTmpBatch=nullptr;
    double* dWave=nullptr;
    double* dReduce=nullptr;
    std::vector<double> hReduce;
    int exactBatchSize = std::max(1, nctrls);

    try {
        cudaCheck(cudaMalloc(&dFwd, static_cast<size_t>(n)*(nsteps+1)*sizeof(cdouble)), "cudaMalloc fwd");
        cudaCheck(cudaMalloc(&dBwd, static_cast<size_t>(n)*(nsteps+1)*sizeof(cdouble)), "cudaMalloc bwd");
        cudaCheck(cudaMalloc(&dTerm, n*sizeof(cdouble)), "cudaMalloc term");
        cudaCheck(cudaMalloc(&dWork, n*sizeof(cdouble)), "cudaMalloc work");
        cudaCheck(cudaMalloc(&dEta, n*sizeof(cdouble)), "cudaMalloc eta");
        cudaCheck(cudaMalloc(&dTermEta, n*sizeof(cdouble)), "cudaMalloc term eta");
        cudaCheck(cudaMalloc(&dTermRho, n*sizeof(cdouble)), "cudaMalloc term rho");
        cudaCheck(cudaMalloc(&dSumRho, n*sizeof(cdouble)), "cudaMalloc sum rho");
        cudaCheck(cudaMalloc(&dWorkEta, n*sizeof(cdouble)), "cudaMalloc work eta");
        cudaCheck(cudaMalloc(&dWorkRho, n*sizeof(cdouble)), "cudaMalloc work rho");
        cudaCheck(cudaMalloc(&dTmp, n*sizeof(cdouble)), "cudaMalloc tmp");
        cudaCheck(cudaMalloc(&dGrad, std::max(1, nctrls*nsteps)*sizeof(cdouble)), "cudaMalloc grad");

        size_t waveElems = static_cast<size_t>(nctrls) * nsteps;
        if (waveElems > 0) {
            cudaCheck(cudaMalloc(&dWave, waveElems * sizeof(double)), "cudaMalloc waveform");
            cudaCheck(cudaMemcpy(dWave, hWave, waveElems * sizeof(double), cudaMemcpyHostToDevice),
                      "copy waveform");
        }

        auto freeBatchBuffers = [&]() {
            if (dEtaBatch) { cudaFree(dEtaBatch); dEtaBatch = nullptr; }
            if (dTermEtaBatch) { cudaFree(dTermEtaBatch); dTermEtaBatch = nullptr; }
            if (dWorkEtaBatch) { cudaFree(dWorkEtaBatch); dWorkEtaBatch = nullptr; }
            if (dTmpBatch) { cudaFree(dTmpBatch); dTmpBatch = nullptr; }
        };
        auto allocateBatchBuffers = [&](int batchSize) -> cudaError_t {
            freeBatchBuffers();
            int elems = checkedBatchElements(n, batchSize, "exact-gradient batch allocation");
            size_t bytes = static_cast<size_t>(elems) * sizeof(cdouble);
            cudaError_t err = cudaMalloc(&dEtaBatch, bytes);
            if (err != cudaSuccess) { freeBatchBuffers(); return err; }
            err = cudaMalloc(&dTermEtaBatch, bytes);
            if (err != cudaSuccess) { freeBatchBuffers(); return err; }
            err = cudaMalloc(&dWorkEtaBatch, bytes);
            if (err != cudaSuccess) { freeBatchBuffers(); return err; }
            err = cudaMalloc(&dTmpBatch, bytes);
            if (err != cudaSuccess) { freeBatchBuffers(); return err; }
            return cudaSuccess;
        };
        while (true) {
            cudaError_t err = allocateBatchBuffers(exactBatchSize);
            if (err == cudaSuccess) break;
            if (exactBatchSize == 1) cudaCheck(err, "cudaMalloc exact-gradient batch buffers");
            exactBatchSize = std::max(1, exactBatchSize / 2);
        }
        int reduceLen = std::max(n, checkedBatchElements(n, exactBatchSize, "exact-gradient reduction"));
        cudaCheck(cudaMalloc(&dReduce, blocksFor(reduceLen)*sizeof(double)), "cudaMalloc reduce");

        cudaCheck(cudaMemcpy(dFwd, hRhoInit.data(), n*sizeof(cdouble), cudaMemcpyHostToDevice), "copy rho_init");
        cudaCheck(cudaMemcpy(dBwd + static_cast<size_t>(n)*nsteps, hRhoTarg.data(), n*sizeof(cdouble), cudaMemcpyHostToDevice), "copy rho_targ");

        // Forward trajectory: rho_{s+1} = P_s rho_s.
        for (int s = 0; s < nsteps; ++s) {
            int driftIndex = s % pack->ndrifts;
            const double* coeff = hWave + static_cast<size_t>(s)*nctrls;
            const double* dCoeff = dWave ? dWave + static_cast<size_t>(s)*nctrls : nullptr;
            propagateState(pack, driftIndex, coeff, dCoeff, false,
                           dFwd + static_cast<size_t>(n)*s,
                           dFwd + static_cast<size_t>(n)*(s+1),
                           hDt[s], dTerm, dWork, dReduce, hReduce);
        }

        // Backward trajectory: lambda_s = P_s^H lambda_{s+1}.
        for (int s = nsteps-1; s >= 0; --s) {
            int driftIndex = s % pack->ndrifts;
            const double* coeff = hWave + static_cast<size_t>(s)*nctrls;
            const double* dCoeff = dWave ? dWave + static_cast<size_t>(s)*nctrls : nullptr;
            propagateState(pack, driftIndex, coeff, dCoeff, true,
                           dBwd + static_cast<size_t>(n)*(s+1),
                           dBwd + static_cast<size_t>(n)*s,
                           -hDt[s], dTerm, dWork, dReduce, hReduce);
        }

        cdouble* dOverlap = dGrad; // reuse first scalar before gradient fill
        dot_store_kernel<<<1, THREADS>>>(dBwd + static_cast<size_t>(n)*nsteps,
                                         dFwd + static_cast<size_t>(n)*nsteps,
                                         n, dOverlap, 0);
        cudaCheck(cudaGetLastError(), "dot overlap");
        cdouble hOverlap;
        cudaCheck(cudaMemcpy(&hOverlap, dOverlap, sizeof(cdouble), cudaMemcpyDeviceToHost), "copy overlap");

        bool wantGrad = (nlhs >= 3);
        if (wantGrad) {
            grumbleGradientMode(gradientMode);
            for (int s = 0; s < nsteps; ++s) {
                int driftIndex = s % pack->ndrifts;
                const double* coeff = hWave + static_cast<size_t>(s)*nctrls;
                const double* dCoeff = dWave ? dWave + static_cast<size_t>(s)*nctrls : nullptr;
                if (isMidpointGradientMode(gradientMode) || isScalarExactGradientMode(gradientMode)) {
                    for (int k = 0; k < nctrls; ++k) {
                        if (isMidpointGradientMode(gradientMode)) {
                            midpoint_state_kernel<<<blocksFor(n), THREADS>>>(
                                dFwd + static_cast<size_t>(n)*s,
                                dFwd + static_cast<size_t>(n)*(s+1), dTmp, n);
                            cudaCheck(cudaGetLastError(), "midpoint_state_kernel");
                            controlAction(pack, k, false, dTmp, dEta);
                            cdouble factor = make_cd(0.0, -hDt[s]);
                            linear_scale_kernel<<<blocksFor(n), THREADS>>>(dEta, dEta, n, factor);
                            cudaCheck(cudaGetLastError(), "midpoint derivative scale");
                        } else {
                            propagateFrechet(pack, driftIndex, coeff, dCoeff, k,
                                             dFwd + static_cast<size_t>(n)*s,
                                             dEta, hDt[s], dTermEta, dTermRho,
                                             dSumRho, dWorkEta, dWorkRho, dTmp,
                                             dReduce, hReduce);
                        }
                        int outIdx = k + s*nctrls;
                        dot_store_kernel<<<1, THREADS>>>(dBwd + static_cast<size_t>(n)*(s+1),
                                                         dEta, n, dGrad, outIdx);
                        cudaCheck(cudaGetLastError(), "dot gradient");
                    }
                } else {
                    for (int firstCtrl = 0; firstCtrl < nctrls; firstCtrl += exactBatchSize) {
                        int nBatch = std::min(exactBatchSize, nctrls - firstCtrl);
                        propagateFrechetBatch(pack, driftIndex, coeff, dCoeff, firstCtrl, nBatch,
                                              dFwd + static_cast<size_t>(n)*s,
                                              dEtaBatch, hDt[s], dTermEtaBatch, dTermRho,
                                              dSumRho, dWorkEtaBatch, dWorkRho, dTmpBatch,
                                              dReduce, hReduce, useImplicitFused);
                        dot_batch_store_kernel<<<nBatch, THREADS>>>(
                            dBwd + static_cast<size_t>(n)*(s+1), dEtaBatch, n,
                            nBatch, dGrad, firstCtrl + s*nctrls, 1);
                        cudaCheck(cudaGetLastError(), "dot batch gradient");
                    }
                }
            }
            int mode = fidelityMode(fidelityType);
            postprocess_gradient_kernel<<<blocksFor(nctrls*nsteps), THREADS>>>(dGrad, nctrls*nsteps, hOverlap, mode);
            cudaCheck(cudaGetLastError(), "postprocess_gradient_kernel");
        }

        int mode = fidelityMode(fidelityType);
        double fidReal;
        if (mode == 0) fidReal = hOverlap.x;
        else if (mode == 1) fidReal = hOverlap.y;
        else fidReal = hOverlap.x*hOverlap.x + hOverlap.y*hOverlap.y;

        // Output 1: traj_data struct.
        const char* trajFields[] = {"forward"};
        mxArray* traj = mxCreateStructMatrix(1, 1, 1, trajFields);
        if (returnForward) {
            std::vector<cdouble> hFwd(static_cast<size_t>(n)*(nsteps+1));
            cudaCheck(cudaMemcpy(hFwd.data(), dFwd, hFwd.size()*sizeof(cdouble), cudaMemcpyDeviceToHost), "copy fwd traj");
            mxSetField(traj, 0, "forward", makeComplexMatrix(n, nsteps+1, hFwd));
        } else {
            mxSetField(traj, 0, "forward", mxCreateDoubleMatrix(0, 0, mxREAL));
        }
        if (nlhs >= 1) plhs[0] = traj; else mxDestroyArray(traj);
        if (nlhs >= 2) plhs[1] = mxCreateDoubleScalar(fidReal);
        if (nlhs >= 3) {
            std::vector<cdouble> hGrad(nctrls*nsteps);
            cudaCheck(cudaMemcpy(hGrad.data(), dGrad, hGrad.size()*sizeof(cdouble), cudaMemcpyDeviceToHost), "copy grad");
            mxArray* grad = mxCreateDoubleMatrix(nctrls, nsteps, mxREAL);
            double* g = mxGetDoubles(grad);
            for (int i = 0; i < nctrls*nsteps; ++i) g[i] = hGrad[i].x;
            plhs[2] = grad;
        }
    } catch (...) {
        if (dFwd) cudaFree(dFwd); if (dBwd) cudaFree(dBwd);
        if (dTerm) cudaFree(dTerm); if (dWork) cudaFree(dWork);
        if (dEta) cudaFree(dEta); if (dTermEta) cudaFree(dTermEta);
        if (dTermRho) cudaFree(dTermRho); if (dSumRho) cudaFree(dSumRho);
        if (dWorkEta) cudaFree(dWorkEta); if (dWorkRho) cudaFree(dWorkRho);
        if (dEtaBatch) cudaFree(dEtaBatch); if (dTermEtaBatch) cudaFree(dTermEtaBatch);
        if (dWorkEtaBatch) cudaFree(dWorkEtaBatch); if (dTmpBatch) cudaFree(dTmpBatch);
        if (dWave) cudaFree(dWave);
        if (dTmp) cudaFree(dTmp); if (dGrad) cudaFree(dGrad); if (dReduce) cudaFree(dReduce);
        throw;
    }

    if (dFwd) cudaFree(dFwd); if (dBwd) cudaFree(dBwd);
    if (dTerm) cudaFree(dTerm); if (dWork) cudaFree(dWork);
    if (dEta) cudaFree(dEta); if (dTermEta) cudaFree(dTermEta);
    if (dTermRho) cudaFree(dTermRho); if (dSumRho) cudaFree(dSumRho);
    if (dWorkEta) cudaFree(dWorkEta); if (dWorkRho) cudaFree(dWorkRho);
    if (dEtaBatch) cudaFree(dEtaBatch); if (dTermEtaBatch) cudaFree(dTermEtaBatch);
    if (dWorkEtaBatch) cudaFree(dWorkEtaBatch); if (dTmpBatch) cudaFree(dTmpBatch);
    if (dWave) cudaFree(dWave);
    if (dTmp) cudaFree(dTmp); if (dGrad) cudaFree(dGrad); if (dReduce) cudaFree(dReduce);
}

// Store a pack in the handle table and lock the MEX file in memory
std::uint64_t registerPack(OperatorPack* pack) {
    std::uint64_t handle = g_nextHandle++;
    if (g_nextHandle == 0) g_nextHandle = 1;
    g_packs[handle] = pack;
    mexLock();
    return handle;
}

// Destroy the pack behind a handle and release it from the table
void destroyHandle(std::uint64_t handle) {
    auto it = g_packs.find(handle);
    if (it != g_packs.end()) {
        destroyPack(it->second);
        g_packs.erase(it);
        if (mexIsLocked()) mexUnlock();
    }
}

// Build a throwaway pack, evaluate once, and destroy it again
void directEvaluate(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {

    // Check consistency
    grumbleDirectArity(nrhs);
    grumbleIntegrator(getString(prhs[7], "integrator"));
    std::string gradientMode = (nrhs >= 9) ? getString(prhs[8], "gradient_mode") : std::string("exact");
    bool returnForward = false;
    if (nrhs >= 10) returnForward = mxIsLogicalScalarTrue(prhs[9]);

    OperatorPack* pack = nullptr;
    try {
        pack = createPackFromInputs(prhs[0], prhs[1], prhs[5]);
        evaluatePack(pack, prhs[2], prhs[3], prhs[4], getString(prhs[6], "fidelity_type"),
                     gradientMode, returnForward, nlhs, plhs);
        destroyPack(pack);
    } catch (...) {
        destroyPack(pack);
        throw;
    }
}

}  // namespace

#endif  // GRAPE_GPU_EVAL_CUH
