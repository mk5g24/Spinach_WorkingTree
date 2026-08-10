/*
 * grape_gpu_propagate.cuh
 *
 * Time stepping.  The state is advanced across one time slice by a
 * reordered Taylor series for exp(-1i*L*dt), scaled down into as many
 * substeps as the norm of L*dt requires, and the exact Frechet derivatives
 * of that step with respect to the control amplitudes are carried alongside
 * it by an augmented recurrence rather than by materialising block matrices.
 *
 * Each recurrence appears twice: once for a single spin system, and once for
 * an ensemble-resident pack that advances every member at the same time.
 * The implicit variants fuse the coupled state and derivative recurrence
 * into a single padded-CSR action.
 *
 * The convergence and substep guards here are numerical statements rather
 * than argument checks, which is why they are not in grape_gpu_grumble.cuh.
 */

#ifndef GRAPE_GPU_PROPAGATE_CUH
#define GRAPE_GPU_PROPAGATE_CUH

#include "grape_gpu_actions.cuh"

namespace {

// Upper bound on the infinity norm of the affine Liouvillian at one time slice
double affineNormUpper(const OperatorPack* pack, int driftIndex, const double* hCoeff, bool adjoint) {
    const CsrDevice& drift = adjoint ? pack->driftsAdj[driftIndex] : pack->drifts[driftIndex];
    const std::vector<CsrDevice>& ctrls = adjoint ? pack->controlsAdj : pack->controls;
    double nrm = drift.infNorm;
    for (int k = 0; k < pack->nctrls; ++k) nrm += std::abs(hCoeff[k]) * ctrls[k].infNorm;
    return nrm;
}

// Advances the state one time slice by a reordered Taylor series for exp(-1i*L*dt)
void propagateState(const OperatorPack* pack, int driftIndex, const double* hCoeff,
                    const double* dCoeff, bool adjoint, const cdouble* dInput, cdouble* dOutput,
                    double dt, cdouble* dTerm, cdouble* dWork,
                    double* dReduce, std::vector<double>& hReduce) {
    const int n = pack->n;
    copyDevice(dInput, dOutput, n);

    double scaling = deviceMaxAbs(dOutput, n, dReduce, hReduce);
    if (scaling == 0.0) return;
    scaleDevice(dOutput, n, 1.0 / scaling);

    double normMat = affineNormUpper(pack, driftIndex, hCoeff, adjoint) * std::abs(dt);
    int nSub = static_cast<int>(std::ceil(normMat / 2.0));
    if (nSub > 10000) throw std::runtime_error("|L*dt| requires more than 10000 substeps.");
    if (nSub < 1) {
        scaleDevice(dOutput, n, scaling);
        return;
    }

    for (int sub = 0; sub < nSub; ++sub) {
        double termMax = copyAndMaxAbs(dOutput, dTerm, n, dReduce, hReduce);
        int iter = 1;
        while (termMax > TAYLOR_TOL) {
            operatorAction(pack, driftIndex, hCoeff, dCoeff, adjoint, dTerm, dWork);
            double a = static_cast<double>(iter);
            cdouble factor = make_cd(0.0, -(dt / static_cast<double>(nSub)) / a);
            linearScaleAxpyDevice(dTerm, dWork, dOutput, n, factor,
                                  "linear_scale_axpy_kernel state");
            termMax = deviceMaxAbs(dTerm, n, dReduce, hReduce);
            ++iter;
            if (iter > MAX_TAYLOR_ITER) {
                throw std::runtime_error("Taylor series failed to converge within MAX_TAYLOR_ITER.");
            }
        }
    }

    scaleDevice(dOutput, n, scaling);
}

// One step of the augmented recurrence that carries a single derivative
void augmentedAction(const OperatorPack* pack, int driftIndex, const double* hCoeff,
                     const double* dCoeff, int ctrlIndex, bool adjoint,
                     const cdouble* eta, const cdouble* rho,
                     cdouble* outEta, cdouble* outRho,
                     cdouble* tmp) {
    operatorAction(pack, driftIndex, hCoeff, dCoeff, adjoint, eta, outEta);
    controlAction(pack, ctrlIndex, adjoint, rho, tmp);
    axpyDevice(outEta, tmp, pack->n);
    operatorAction(pack, driftIndex, hCoeff, dCoeff, adjoint, rho, outRho);
}

// One step of the augmented recurrence for a block of control derivatives
void augmentedActionBatch(const OperatorPack* pack, int driftIndex, const double* hCoeff,
                          const double* dCoeff, int firstCtrl, int nvec, bool adjoint,
                          const cdouble* etaBatch, const cdouble* rho,
                          cdouble* outEtaBatch, cdouble* outRho,
                          cdouble* tmpBatch) {
    operatorActionBatch(pack, driftIndex, hCoeff, dCoeff, adjoint, etaBatch, outEtaBatch, nvec);
    controlActionBatch(pack, firstCtrl, nvec, adjoint, rho, tmpBatch);
    axpyDevice(outEtaBatch, tmpBatch, checkedBatchElements(pack->n, nvec, "batched augmented eta"));
    operatorAction(pack, driftIndex, hCoeff, dCoeff, adjoint, rho, outRho);
}

// One step of the coupled state and derivative recurrence, fused where possible
void implicitFrechetActionBatch(const OperatorPack* pack, int driftIndex, const double* hCoeff,
                                const double* dCoeff, int firstCtrl, int nvec, bool adjoint,
                                const cdouble* etaBatch, const cdouble* rho,
                                cdouble* outEtaBatch, cdouble* outRho,
                                cdouble* tmpBatch) {
    const FusedCsrDevice& F = adjoint ? pack->fusedAdj : pack->fused;
    if (F.rowPtr != nullptr && F.structurallyComplete && F.driftValues != nullptr &&
        F.controlValues != nullptr && dCoeff != nullptr) {
        fusedImplicitFrechetActionBatch(F, driftIndex, dCoeff, firstCtrl, nvec,
                                        etaBatch, rho, outEtaBatch, outRho);
        return;
    }
    augmentedActionBatch(pack, driftIndex, hCoeff, dCoeff, firstCtrl, nvec, adjoint,
                         etaBatch, rho, outEtaBatch, outRho, tmpBatch);
}

// Advances the state and one exact Frechet derivative over one time slice
void propagateFrechet(const OperatorPack* pack, int driftIndex, const double* hCoeff,
                      const double* dCoeff, int ctrlIndex, const cdouble* dRhoIn, cdouble* dEtaOut,
                      double dt, cdouble* dTermEta, cdouble* dTermRho,
                      cdouble* dSumRho, cdouble* dWorkEta, cdouble* dWorkRho,
                      cdouble* dTmp, double* dReduce, std::vector<double>& hReduce) {
    const int n = pack->n;
    fillDevice(dEtaOut, n);
    copyDevice(dRhoIn, dSumRho, n);

    double scaling = deviceMaxAbs(dSumRho, n, dReduce, hReduce);
    if (scaling == 0.0) return;
    scaleDevice(dSumRho, n, 1.0 / scaling);

    double normMat = affineNormUpper(pack, driftIndex, hCoeff, false) + pack->controls[ctrlIndex].infNorm;
    normMat *= std::abs(dt);
    int nSub = static_cast<int>(std::ceil(normMat / 2.0));
    if (nSub > 10000) throw std::runtime_error("augmented |L*dt| requires more than 10000 substeps.");
    if (nSub < 1) {
        scaleDevice(dEtaOut, n, scaling);
        return;
    }

    for (int sub = 0; sub < nSub; ++sub) {
        double termMax = copyPairAndMaxAbs(dEtaOut, dTermEta, n,
                                           dSumRho, dTermRho, n,
                                           dReduce, hReduce);
        int iter = 1;
        while (termMax > TAYLOR_TOL) {

            augmentedAction(pack, driftIndex, hCoeff, dCoeff, ctrlIndex, false,
                            dTermEta, dTermRho, dWorkEta, dWorkRho, dTmp);
            double a = static_cast<double>(iter);
            cdouble factor = make_cd(0.0, -(dt / static_cast<double>(nSub)) / a);
            linearScaleAxpyDevice(dTermEta, dWorkEta, dEtaOut, n, factor,
                                  "linear_scale_axpy_kernel eta");
            linearScaleAxpyDevice(dTermRho, dWorkRho, dSumRho, n, factor,
                                  "linear_scale_axpy_kernel rho");
            termMax = deviceMaxAbsPair(dTermEta, n, dTermRho, n,
                                       dReduce, hReduce);
            ++iter;
            if (iter > MAX_TAYLOR_ITER) {
                throw std::runtime_error("augmented Taylor series failed to converge within MAX_TAYLOR_ITER.");
            }
        }
    }

    scaleDevice(dEtaOut, n, scaling);
}

// Advances the state and a block of exact Frechet derivatives over one time slice
void propagateFrechetBatch(const OperatorPack* pack, int driftIndex, const double* hCoeff,
                           const double* dCoeff, int firstCtrl, int nvec, const cdouble* dRhoIn,
                           cdouble* dEtaBatchOut, double dt,
                           cdouble* dTermEtaBatch, cdouble* dTermRho,
                           cdouble* dSumRho, cdouble* dWorkEtaBatch,
                           cdouble* dWorkRho, cdouble* dTmpBatch,
                           double* dReduce, std::vector<double>& hReduce,
                           bool useImplicitFused = false) {
    const int n = pack->n;
    const int batchElems = checkedBatchElements(n, nvec, "batched Frechet state block");
    fillDevice(dEtaBatchOut, batchElems);
    copyDevice(dRhoIn, dSumRho, n);

    double scaling = deviceMaxAbs(dSumRho, n, dReduce, hReduce);
    if (scaling == 0.0) return;
    scaleDevice(dSumRho, n, 1.0 / scaling);

    double ctrlNorm = 0.0;
    for (int j = 0; j < nvec; ++j) {
        ctrlNorm = std::max(ctrlNorm, pack->controls[firstCtrl + j].infNorm);
    }
    double normMat = (affineNormUpper(pack, driftIndex, hCoeff, false) + ctrlNorm) * std::abs(dt);
    int nSub = static_cast<int>(std::ceil(normMat / 2.0));
    if (nSub > 10000) throw std::runtime_error("batched augmented |L*dt| requires more than 10000 substeps.");
    if (nSub < 1) {
        scaleDevice(dEtaBatchOut, batchElems, scaling);
        return;
    }

    for (int sub = 0; sub < nSub; ++sub) {
        double termMax = copyPairAndMaxAbs(dEtaBatchOut, dTermEtaBatch,
                                           batchElems, dSumRho, dTermRho, n,
                                           dReduce, hReduce);
        int iter = 1;
        while (termMax > TAYLOR_TOL) {

            if (useImplicitFused) {
                implicitFrechetActionBatch(pack, driftIndex, hCoeff, dCoeff, firstCtrl, nvec, false,
                                           dTermEtaBatch, dTermRho, dWorkEtaBatch, dWorkRho,
                                           dTmpBatch);
            } else {
                augmentedActionBatch(pack, driftIndex, hCoeff, dCoeff, firstCtrl, nvec, false,
                                     dTermEtaBatch, dTermRho, dWorkEtaBatch, dWorkRho,
                                     dTmpBatch);
            }
            double a = static_cast<double>(iter);
            cdouble factor = make_cd(0.0, -(dt / static_cast<double>(nSub)) / a);
            linearScaleAxpyDevice(dTermEtaBatch, dWorkEtaBatch, dEtaBatchOut,
                                  batchElems, factor,
                                  "linear_scale_axpy_kernel batched eta");
            linearScaleAxpyDevice(dTermRho, dWorkRho, dSumRho, n, factor,
                                  "linear_scale_axpy_kernel batched rho");
            termMax = deviceMaxAbsPair(dTermEtaBatch, batchElems, dTermRho, n,
                                       dReduce, hReduce);
            ++iter;
            if (iter > MAX_TAYLOR_ITER) {
                throw std::runtime_error("batched augmented Taylor series failed to converge within MAX_TAYLOR_ITER.");
            }
        }
    }

    scaleDevice(dEtaBatchOut, batchElems, scaling);
}

// Largest affine Liouvillian norm bound over all ensemble members
double affineNormUpperEnsemble(const OperatorPack* pack, int driftIndex,
                               const double* hCoeff, bool adjoint) {
    const std::vector<CsrDevice>& drifts = adjoint ? pack->driftsAdj : pack->drifts;
    const std::vector<CsrDevice>& ctrls = adjoint ? pack->controlsAdj : pack->controls;
    double ctrlNorm = 0.0;
    for (int k = 0; k < pack->nctrls; ++k) ctrlNorm += std::abs(hCoeff[k]) * ctrls[k].infNorm;
    double nrm = 0.0;
    for (int sys = 0; sys < pack->nsystems; ++sys) {
        int idx = sys * pack->ndrifts + driftIndex;
        nrm = std::max(nrm, drifts[idx].infNorm + ctrlNorm);
    }
    return nrm;
}

// Applies the per-member affine Liouvillian to one vector per member
void operatorActionEnsemble(const OperatorPack* pack, int driftIndex, const double* dCoeff,
                            bool adjoint, const cdouble* xSystems, cdouble* ySystems) {
    const FusedCsrDevice& F = adjoint ? pack->fusedAdj : pack->fused;
    if (F.rowPtr == nullptr || F.driftValues == nullptr || F.controlValues == nullptr) {
        throw std::runtime_error("ensemble-resident evaluation currently requires a padded union-pattern fused CSR pack.");
    }
    fusedAffineActionEnsemble(F, driftIndex, dCoeff, xSystems, ySystems);
}

// Applies the per-member affine Liouvillian to a batch of vectors per member
void operatorActionEnsembleBatch(const OperatorPack* pack, int driftIndex, const double* dCoeff,
                                 bool adjoint, const cdouble* xBatch, cdouble* yBatch,
                                 int nvecPerSystem) {
    const FusedCsrDevice& F = adjoint ? pack->fusedAdj : pack->fused;
    if (F.rowPtr == nullptr || F.driftValues == nullptr || F.controlValues == nullptr) {
        throw std::runtime_error("ensemble-resident batched evaluation currently requires a padded union-pattern fused CSR pack.");
    }
    fusedAffineActionEnsembleBatch(F, driftIndex, dCoeff, xBatch, yBatch, nvecPerSystem);
}

// Applies a block of control operators to a batch of vectors per member
void controlActionEnsembleBatch(const OperatorPack* pack, int firstCtrl, int nvecPerSystem,
                                bool adjoint, const cdouble* xSystems, cdouble* yBatch) {
    const FusedCsrDevice& F = adjoint ? pack->fusedAdj : pack->fused;
    if (F.rowPtr == nullptr || F.controlValues == nullptr) {
        throw std::runtime_error("ensemble-resident batched control action requires a padded union-pattern fused CSR pack.");
    }
    fusedControlActionEnsembleBatch(F, firstCtrl, nvecPerSystem, xSystems, yBatch);
}

// Advances every ensemble member one time slice by the reordered Taylor series
void propagateStateEnsemble(const OperatorPack* pack, int driftIndex, const double* hCoeff,
                            const double* dCoeff, bool adjoint,
                            const cdouble* dInput, cdouble* dOutput,
                            double dt, cdouble* dTerm, cdouble* dWork,
                            double* dReduce, std::vector<double>& hReduce,
                            cdouble* dAffineValues = nullptr,
                            EvalStats* stats = nullptr) {
    const int stateElems = checkedBatchElements(pack->n, pack->nsystems, "ensemble state block");
    copyDevice(dInput, dOutput, stateElems);

    double scaling = deviceMaxAbs(dOutput, stateElems, dReduce, hReduce);
    if (scaling == 0.0) return;
    scaleDevice(dOutput, stateElems, 1.0 / scaling);

    double normMat = affineNormUpperEnsemble(pack, driftIndex, hCoeff, adjoint) * std::abs(dt);
    int nSub = static_cast<int>(std::ceil(normMat / 2.0));
    if (nSub > 10000) throw std::runtime_error("ensemble |L*dt| requires more than 10000 substeps.");
    if (nSub < 1) {
        scaleDevice(dOutput, stateElems, scaling);
        return;
    }

    const FusedCsrDevice& F = adjoint ? pack->fusedAdj : pack->fused;
    const bool usePreassembledAffine = (dAffineValues != nullptr && F.rowPtr != nullptr &&
                                        F.driftValues != nullptr && F.controlValues != nullptr &&
                                        dCoeff != nullptr);
    if (usePreassembledAffine) {
        assembleAffineEnsemble(F, driftIndex, dCoeff, dAffineValues);
        addAffineAssemblyStats(stats, F);
    }

    for (int sub = 0; sub < nSub; ++sub) {
        double termMax = copyAndMaxAbs(dOutput, dTerm, stateElems,
                                       dReduce, hReduce);
        int iter = 1;
        while (termMax > TAYLOR_TOL) {
            if (usePreassembledAffine) {
                affineActionEnsemble(F, dAffineValues, dTerm, dWork);
                addAffineSpmvStats(stats, F, 1);
            } else {
                operatorActionEnsemble(pack, driftIndex, dCoeff, adjoint, dTerm, dWork);
            }
            double a = static_cast<double>(iter);
            cdouble factor = make_cd(0.0, -(dt / static_cast<double>(nSub)) / a);
            linearScaleAxpyDevice(dTerm, dWork, dOutput, stateElems, factor,
                                  "linear_scale_axpy_kernel ensemble state");
            termMax = deviceMaxAbs(dTerm, stateElems, dReduce, hReduce);
            ++iter;
            if (iter > MAX_TAYLOR_ITER) {
                throw std::runtime_error("ensemble Taylor series failed to converge within MAX_TAYLOR_ITER.");
            }
        }
    }

    scaleDevice(dOutput, stateElems, scaling);
}

// One step of the augmented ensemble recurrence for a block of derivatives
void augmentedActionEnsembleBatch(const OperatorPack* pack, int driftIndex,
                                  const double* dCoeff, int firstCtrl, int nvecPerSystem,
                                  bool adjoint, const cdouble* etaBatch,
                                  const cdouble* rhoSystems, cdouble* outEtaBatch,
                                  cdouble* outRhoSystems, cdouble* tmpBatch) {
    operatorActionEnsembleBatch(pack, driftIndex, dCoeff, adjoint, etaBatch, outEtaBatch, nvecPerSystem);
    controlActionEnsembleBatch(pack, firstCtrl, nvecPerSystem, adjoint, rhoSystems, tmpBatch);
    int totalEta = checkedBatchElements(checkedBatchElements(pack->n, pack->nsystems, "ensemble eta systems"),
                                        nvecPerSystem, "ensemble eta batch");
    axpyDevice(outEtaBatch, tmpBatch, totalEta);
    operatorActionEnsemble(pack, driftIndex, dCoeff, adjoint, rhoSystems, outRhoSystems);
}

// One step of the augmented ensemble recurrence using preassembled affine values
void augmentedActionEnsembleBatchAffine(const OperatorPack* pack, const FusedCsrDevice& F,
                                        const cdouble* affineValues, int firstCtrl,
                                        int nvecPerSystem, bool adjoint,
                                        const cdouble* etaBatch,
                                        const cdouble* rhoSystems,
                                        cdouble* outEtaBatch,
                                        cdouble* outRhoSystems,
                                        cdouble* tmpBatch) {
    affineActionEnsembleBatch(F, affineValues, etaBatch, outEtaBatch, nvecPerSystem);
    controlActionEnsembleBatch(pack, firstCtrl, nvecPerSystem, adjoint, rhoSystems, tmpBatch);
    int totalEta = checkedBatchElements(checkedBatchElements(pack->n, pack->nsystems, "ensemble eta systems"),
                                        nvecPerSystem, "ensemble eta batch");
    axpyDevice(outEtaBatch, tmpBatch, totalEta);
    affineActionEnsemble(F, affineValues, rhoSystems, outRhoSystems);
}

// One step of the coupled ensemble state and derivative recurrence
void implicitFrechetActionEnsembleBatch(const OperatorPack* pack, int driftIndex,
                                        const double* dCoeff, int firstCtrl,
                                        int nvecPerSystem, bool adjoint,
                                        const cdouble* etaBatch,
                                        const cdouble* rhoSystems,
                                        cdouble* outEtaBatch,
                                        cdouble* outRhoSystems,
                                        cdouble* tmpBatch) {
    const FusedCsrDevice& F = adjoint ? pack->fusedAdj : pack->fused;
    if (F.rowPtr != nullptr && F.structurallyComplete && F.driftValues != nullptr &&
        F.controlValues != nullptr && dCoeff != nullptr) {
        fusedImplicitFrechetActionEnsembleBatch(F, driftIndex, dCoeff, firstCtrl,
                                                nvecPerSystem, etaBatch, rhoSystems,
                                                outEtaBatch, outRhoSystems);
        return;
    }
    augmentedActionEnsembleBatch(pack, driftIndex, dCoeff, firstCtrl, nvecPerSystem,
                                 adjoint, etaBatch, rhoSystems, outEtaBatch,
                                 outRhoSystems, tmpBatch);
}

// Advances every member state and its derivative block over one time slice
void propagateFrechetEnsembleBatch(const OperatorPack* pack, int driftIndex,
                                   const double* hCoeff, const double* dCoeff,
                                   int firstCtrl, int nvecPerSystem,
                                   const cdouble* dRhoSystemsIn,
                                   cdouble* dEtaBatchOut, double dt,
                                   cdouble* dTermEtaBatch, cdouble* dTermRhoSystems,
                                   cdouble* dSumRhoSystems, cdouble* dWorkEtaBatch,
                                   cdouble* dWorkRhoSystems, cdouble* dTmpBatch,
                                   double* dReduce, std::vector<double>& hReduce,
                                   bool useImplicitFused = false,
                                   cdouble* dAffineValues = nullptr,
                                   EvalStats* stats = nullptr) {
    const int stateElems = checkedBatchElements(pack->n, pack->nsystems, "ensemble Frechet states");
    const int batchElems = checkedBatchElements(stateElems, nvecPerSystem, "ensemble Frechet state-control block");
    fillDevice(dEtaBatchOut, batchElems);
    copyDevice(dRhoSystemsIn, dSumRhoSystems, stateElems);

    double scaling = deviceMaxAbs(dSumRhoSystems, stateElems, dReduce, hReduce);
    if (scaling == 0.0) return;
    scaleDevice(dSumRhoSystems, stateElems, 1.0 / scaling);

    double ctrlNorm = 0.0;
    for (int j = 0; j < nvecPerSystem; ++j) {
        ctrlNorm = std::max(ctrlNorm, pack->controls[firstCtrl + j].infNorm);
    }
    double normMat = (affineNormUpperEnsemble(pack, driftIndex, hCoeff, false) + ctrlNorm) * std::abs(dt);
    int nSub = static_cast<int>(std::ceil(normMat / 2.0));
    if (nSub > 10000) throw std::runtime_error("ensemble augmented |L*dt| requires more than 10000 substeps.");
    if (nSub < 1) {
        scaleDevice(dEtaBatchOut, batchElems, scaling);
        return;
    }

    const FusedCsrDevice& F = pack->fused;
    const bool usePreassembledAffine = (dAffineValues != nullptr &&
                                        F.rowPtr != nullptr && F.driftValues != nullptr &&
                                        F.controlValues != nullptr && dCoeff != nullptr);
    if (usePreassembledAffine) {
        assembleAffineEnsemble(F, driftIndex, dCoeff, dAffineValues);
        addAffineAssemblyStats(stats, F);
    }

    for (int sub = 0; sub < nSub; ++sub) {
        double termMax = copyPairAndMaxAbs(dEtaBatchOut, dTermEtaBatch,
                                           batchElems, dSumRhoSystems,
                                           dTermRhoSystems, stateElems,
                                           dReduce, hReduce);
        int iter = 1;
        while (termMax > TAYLOR_TOL) {

            if (useImplicitFused && usePreassembledAffine) {
                implicitFrechetActionEnsembleBatchAffine(F, dAffineValues, firstCtrl,
                                                         nvecPerSystem, dTermEtaBatch,
                                                         dTermRhoSystems, dWorkEtaBatch,
                                                         dWorkRhoSystems);
                addImplicitFrechetStats(stats, F, nvecPerSystem);
            } else if (usePreassembledAffine) {
                augmentedActionEnsembleBatchAffine(pack, F, dAffineValues, firstCtrl,
                                                   nvecPerSystem, false, dTermEtaBatch,
                                                   dTermRhoSystems, dWorkEtaBatch,
                                                   dWorkRhoSystems, dTmpBatch);
                addAffineSpmvStats(stats, F, nvecPerSystem + 1);
            } else if (useImplicitFused) {
                implicitFrechetActionEnsembleBatch(pack, driftIndex, dCoeff, firstCtrl, nvecPerSystem,
                                                   false, dTermEtaBatch, dTermRhoSystems,
                                                   dWorkEtaBatch, dWorkRhoSystems, dTmpBatch);
            } else {
                augmentedActionEnsembleBatch(pack, driftIndex, dCoeff, firstCtrl, nvecPerSystem,
                                             false, dTermEtaBatch, dTermRhoSystems,
                                             dWorkEtaBatch, dWorkRhoSystems, dTmpBatch);
            }
            double a = static_cast<double>(iter);
            cdouble factor = make_cd(0.0, -(dt / static_cast<double>(nSub)) / a);
            linearScaleAxpyDevice(dTermEtaBatch, dWorkEtaBatch, dEtaBatchOut,
                                  batchElems, factor,
                                  "linear_scale_axpy_kernel ensemble eta");
            linearScaleAxpyDevice(dTermRhoSystems, dWorkRhoSystems,
                                  dSumRhoSystems, stateElems, factor,
                                  "linear_scale_axpy_kernel ensemble rho");
            termMax = deviceMaxAbsPair(dTermEtaBatch, batchElems,
                                       dTermRhoSystems, stateElems,
                                       dReduce, hReduce);
            ++iter;
            if (iter > MAX_TAYLOR_ITER) {
                throw std::runtime_error("ensemble augmented Taylor series failed to converge within MAX_TAYLOR_ITER.");
            }
        }
    }

    scaleDevice(dEtaBatchOut, batchElems, scaling);
}

}  // namespace

#endif  // GRAPE_GPU_PROPAGATE_CUH
