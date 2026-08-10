/*
 * grape_liouv_gpu_cuda_mex.cu
 *
 * CUDA/MEX backend for Liouville-space GRAPE in Spinach-style state-vector
 * formalisms.  This file implements the first production-oriented GPU path:
 *
 *   1. persistent operator packs resident on the GPU;
 *   2. CUDA reordered-Taylor exp(-1i*L*dt) state actions with parity intent
 *      against Spinach step.m's GPU/state-vector branch;
 *   3. forward/backward GRAPE trajectories with a fast midpoint gradient;
 *   4. exact Frechet derivatives implemented as augmented state-vector
 *      recurrences rather than materialised block matrices; the optional
 *      exact_implicit mode fuses the coupled rho/eta recurrence into one
 *      padded-CSR action when the union-pattern pack is available.
 *
 * The safe C++ stub in grape_liouv_gpu_mex.cpp remains the default build on
 * systems without CUDA.  Build this backend from MATLAB with
 *
 *   compile_grape_gpu('default')
 *
 * which produces grape_liouv_gpu_mex.<mexext>.  The MATLAB wrapper only trusts
 * a MEX file that reports cudaBackendReady=true from the capabilities query.
 *
 * This file holds only the mexFunction command dispatcher.  The backend is
 * split by concern across five headers, included below in dependency order:
 *
 *   grape_gpu_types.cuh   - storage layouts, constants, arithmetic helpers
 *   grape_gpu_grumble.cuh - argument checking, and the readers that use it
 *   grape_gpu_kernels.cuh - CUDA kernels
 *   grape_gpu_stats.cuh   - work accounting for throughput reporting
 *   grape_gpu_actions.cuh - kernel launchers, one operator application each
 *   grape_gpu_propagate.cuh - Taylor time stepping and Frechet recurrences
 *   grape_gpu_packs.cuh   - persistent GPU operator pack construction
 *   grape_gpu_eval.cuh    - fidelity, gradient, and trajectory evaluation
 *
 * They are headers rather than separately compiled translation units on
 * purpose: the backend stays one translation unit, so every cross-unit call
 * remains inlinable and the generated code is unchanged by the split.
 *
 */

#include "grape_gpu_types.cuh"
#include "grape_gpu_grumble.cuh"
#include "grape_gpu_kernels.cuh"
#include "grape_gpu_stats.cuh"
#include "grape_gpu_actions.cuh"
#include "grape_gpu_propagate.cuh"
#include "grape_gpu_packs.cuh"
#include "grape_gpu_eval.cuh"

// Mex Function wrapper, assures proper communication between Host and Device
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {

    // Memory cleanup at end of calculations
    static bool registered = false;
    if (!registered) {
        mexAtExit(atExitCleanup);
        registered = true;
    }

    try {
        if (nrhs >= 1 && mxIsChar(prhs[0])) {
            
            // Report what this backend implements
            if (isCommand(prhs[0], "capabilities")) {
                grumbleCommandArity(true, "'capabilities'", nlhs <= 1, "one struct");
                plhs[0] = makeCapabilities();
                return;
            }
            // Return current version of the CUDA Grape script
            if (isCommand(prhs[0], "version")) {
                grumbleCommandArity(true, "'version'", nlhs <= 1, "one scalar");
                plhs[0] = mxCreateDoubleScalar(11.0);
                return;
            }
            // Pack the system
            if (isCommand(prhs[0], "pack")) {
                grumbleCommandArity(nrhs == 4, "'pack', drifts, controls, dt",
                                    nlhs <= 1, "one uint64 handle");
                OperatorPack* pack = createPackFromInputs(prhs[1], prhs[2], prhs[3]);
                plhs[0] = makeHandle(registerPack(pack));
                return;
            }
            // Pack an ensemble of systems
            if (isCommand(prhs[0], "pack_ensemble")) {
                grumbleCommandArity(nrhs == 4, "'pack_ensemble', ensemble_drifts, controls, dt",
                                    nlhs <= 1, "one uint64 handle");
                OperatorPack* pack = createEnsemblePackFromInputs(prhs[1], prhs[2], prhs[3]);
                plhs[0] = makeHandle(registerPack(pack));
                return;
            }
            // Clear all handles 
            if (isCommand(prhs[0], "destroy")) {
                grumbleCommandArity(nrhs == 2, "'destroy', handle", nlhs == 0, "nothing");
                destroyHandle(getHandle(prhs[1]));
                return;
            }
            // Clear all packs
            if (isCommand(prhs[0], "clear")) {
                clearAllPacks();
                return;
            }
            // Actual GRAPE algorithm
            if (isCommand(prhs[0], "eval")) {
                grumbleCommandArity(nrhs >= 6,
                    "'eval', handle, waveform, rho_init, rho_targ, fidelity_type [, gradient_mode] [, return_forward]",
                    nlhs <= 3, "trajectory data, fidelity, and gradient");
                OperatorPack* pack = requirePack(getHandle(prhs[1]));
                std::string gradientMode = (nrhs >= 7) ? getString(prhs[6], "gradient_mode") : std::string("exact");
                bool returnForward = (nrhs >= 8) ? mxIsLogicalScalarTrue(prhs[7]) : false;
                evaluatePack(pack, prhs[2], prhs[3], prhs[4], getString(prhs[5], "fidelity_type"),
                             gradientMode, returnForward, nlhs, plhs);
                return;
            }
            throw std::runtime_error("unknown grape_liouv_gpu_mex command.");
        }

        directEvaluate(nlhs, plhs, nrhs, prhs);
    } catch (const std::exception& e) {
        fail("GRAPE:LiouvGPU:CudaBackendError", e.what());
    }
}
