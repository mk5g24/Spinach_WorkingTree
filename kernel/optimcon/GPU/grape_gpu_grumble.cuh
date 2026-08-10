/*
 * grape_gpu_grumble.cuh
 *
 * Argument checking at the MATLAB boundary, and the small readers that
 * apply it.
 *
 * Everything that arrives from MATLAB is checked here and nowhere else, so
 * that the working code downstream can read as arithmetic rather than as a
 * sequence of guards.  This follows the Spinach convention: a function that
 * takes user input calls its grumbler on the first line and then trusts what
 * it was given.
 *
 * Checks that are NOT here, on purpose:
 *
 *   - resource limits that depend on the data rather than on its type, such
 *     as 32-bit index range and launch-size overflow, stay next to the
 *     allocation or launch they protect, because they can only be evaluated
 *     there and the failure mode is silent memory corruption;
 *
 *   - numerical guards, such as Taylor series convergence and substep counts,
 *     which are statements about the physics rather than about the arguments;
 *
 *   - invariants between subfunctions of this backend.  Those subfunctions
 *     are not reachable from MATLAB, their inputs have already been through
 *     a grumbler, and a check that cannot fire is noise.
 */

#ifndef GRAPE_GPU_GRUMBLE_CUH
#define GRAPE_GPU_GRUMBLE_CUH

#include "grape_gpu_types.cuh"

namespace {

// Complains unless the argument is a character array
void grumbleString(const mxArray* arg, const char* name) {
    if (!mxIsChar(arg)) {
        throw std::runtime_error(std::string(name) + " must be a character vector.");
    }
}

// Complains unless the argument is a scalar uint64 operator pack handle
void grumbleHandle(const mxArray* arg) {
    if (!mxIsUint64(arg) || mxGetNumberOfElements(arg) != 1) {
        throw std::runtime_error("operator-pack handle must be a scalar uint64.");
    }
}

// Complains unless the argument is a square double sparse operator, and returns its dimension
int grumbleOperator(const mxArray* A, const char* what) {
    if (A == nullptr) {
        throw std::runtime_error(std::string(what) + " is empty.");
    }
    if (!mxIsSparse(A) || !mxIsDouble(A)) {
        throw std::runtime_error("all operators must be double-precision sparse matrices.");
    }
    mwSize m = mxGetM(A), n = mxGetN(A);
    if (m != n || m > static_cast<mwSize>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("operators must be square and fit 32-bit CSR indices.");
    }
    if (mxGetJc(A)[n] > static_cast<mwSize>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("operator nnz exceeds 32-bit CSR index range.");
    }
    return static_cast<int>(n);
}

// Complains unless the time grid is a non-empty real double vector
void grumbleTimeGrid(const mxArray* dtArg) {
    if (!mxIsDouble(dtArg) || mxIsComplex(dtArg) || mxGetNumberOfElements(dtArg) < 1) {
        throw std::runtime_error("dt must be a real double vector.");
    }
}

// Complains unless the pack inputs are one drift set, one control set, and a time grid
void grumblePackInputs(const mxArray* driftsArg, const mxArray* controlsArg, const mxArray* dtArg) {
    if (!mxIsCell(driftsArg) || mxGetNumberOfElements(driftsArg) < 1) {
        throw std::runtime_error("drifts must be a non-empty cell array.");
    }
    if (!mxIsCell(controlsArg)) {
        throw std::runtime_error("controls must be a cell array.");
    }
    grumbleTimeGrid(dtArg);

    int ndrifts = static_cast<int>(mxGetNumberOfElements(driftsArg));
    int n = grumbleOperator(mxGetCell(driftsArg, 0), "drift matrix");
    for (int i = 1; i < ndrifts; ++i) {
        if (grumbleOperator(mxGetCell(driftsArg, i), "drift matrix") != n) {
            throw std::runtime_error("all drift matrices must have the same dimension.");
        }
    }
    int nctrls = static_cast<int>(mxGetNumberOfElements(controlsArg));
    for (int k = 0; k < nctrls; ++k) {
        if (grumbleOperator(mxGetCell(controlsArg, k), "control matrix") != n) {
            throw std::runtime_error("all control matrices must match drift dimension.");
        }
    }
}

// Complains unless the ensemble pack inputs hold one identically shaped drift set per member
void grumbleEnsembleInputs(const mxArray* driftsEnsembleArg, const mxArray* controlsArg,
                           const mxArray* dtArg) {
    if (!mxIsCell(driftsEnsembleArg) || mxGetNumberOfElements(driftsEnsembleArg) < 1) {
        throw std::runtime_error("ensemble drifts must be a non-empty cell array of drift cells or sparse matrices.");
    }
    if (!mxIsCell(controlsArg)) {
        throw std::runtime_error("controls must be a cell array.");
    }
    grumbleTimeGrid(dtArg);

    const mxArray* firstMember = mxGetCell(driftsEnsembleArg, 0);
    if (firstMember == nullptr) {
        throw std::runtime_error("ensemble drift member 1 is empty.");
    }
    int ndrifts = mxIsCell(firstMember) ? static_cast<int>(mxGetNumberOfElements(firstMember)) : 1;
    if (ndrifts < 1) {
        throw std::runtime_error("ensemble drift members must contain at least one drift matrix.");
    }

    int nsystems = static_cast<int>(mxGetNumberOfElements(driftsEnsembleArg));
    int n = 0;
    for (int sys = 0; sys < nsystems; ++sys) {
        const mxArray* member = mxGetCell(driftsEnsembleArg, sys);
        if (member == nullptr) {
            throw std::runtime_error("ensemble drift member is empty.");
        }
        int memberDrifts = mxIsCell(member) ? static_cast<int>(mxGetNumberOfElements(member)) : 1;
        if (memberDrifts != ndrifts) {
            throw std::runtime_error("all ensemble members must have the same number of drift matrices.");
        }
        for (int i = 0; i < ndrifts; ++i) {
            const mxArray* A = mxIsCell(member) ? mxGetCell(member, i) : member;
            int thisN = grumbleOperator(A, "ensemble drift matrix");
            if (sys == 0 && i == 0) n = thisN;
            if (thisN != n) {
                throw std::runtime_error("all ensemble drift matrices must have the same dimension.");
            }
        }
    }
    int nctrls = static_cast<int>(mxGetNumberOfElements(controlsArg));
    for (int k = 0; k < nctrls; ++k) {
        if (grumbleOperator(mxGetCell(controlsArg, k), "control matrix") != n) {
            throw std::runtime_error("all control matrices must match ensemble drift dimension.");
        }
    }
}

// Complains unless the waveform is a real dense matrix of size n_controls by n_steps
void grumbleWaveform(const mxArray* waveformArg, const OperatorPack* pack) {
    if (!mxIsDouble(waveformArg) || mxIsComplex(waveformArg) || mxIsSparse(waveformArg)) {
        throw std::runtime_error("waveform must be a real dense double matrix.");
    }
    if (mxGetM(waveformArg) != static_cast<mwSize>(pack->nctrls) ||
        mxGetN(waveformArg) != static_cast<mwSize>(pack->nsteps)) {
        throw std::runtime_error("waveform dimensions must be [n_controls x n_steps].");
    }
}

// Complains unless the state vector has the dimension of the operators
void grumbleStateVector(const mxArray* arg, const char* name, int expectedN) {
    if (!mxIsDouble(arg) || mxGetNumberOfElements(arg) != static_cast<mwSize>(expectedN)) {
        throw std::runtime_error(std::string(name) + " must be a double vector matching operator dimension.");
    }
}

// Complains unless the state vectors are common to the ensemble or one per member
void grumbleEnsembleVectors(const mxArray* arg, const char* name, int n, int nsystems) {
    if (!mxIsDouble(arg)) {
        throw std::runtime_error(std::string(name) + " must be a common vector of length n or an n-by-ensemble double array.");
    }
    mwSize elems = mxGetNumberOfElements(arg);
    if (elems == static_cast<mwSize>(n)) return;
    if (mxIsSparse(arg)) {
        if (mxGetM(arg) != static_cast<mwSize>(n) || mxGetN(arg) != static_cast<mwSize>(nsystems)) {
            throw std::runtime_error(std::string(name) + " sparse ensemble array must have size n-by-ensemble.");
        }
        return;
    }
    if (mxGetM(arg) != static_cast<mwSize>(n) || mxGetN(arg) != static_cast<mwSize>(nsystems)) {
        throw std::runtime_error(std::string(name) + " ensemble array must have size n-by-ensemble.");
    }
}

// Complains unless the gradient mode is one this backend implements
void grumbleGradientMode(const std::string& gradientMode) {
    if (!isExactGradientMode(gradientMode) && !isMidpointGradientMode(gradientMode)) {
        throw std::runtime_error("gradient mode must be 'exact'/'frechet', 'exact_implicit'/'frechet_implicit', 'exact_scalar'/'frechet_scalar', or 'midpoint'.");
    }
}

// Complains unless the ensemble evaluation request is one this backend implements
void grumbleEnsembleRequest(const std::string& gradientMode, bool returnForward) {
    if (!isExactGradientMode(gradientMode) || isScalarExactGradientMode(gradientMode)) {
        throw std::runtime_error("ensemble-resident evaluation currently supports batched 'exact'/'frechet' and 'exact_implicit'/'frechet_implicit' gradient modes.");
    }
    if (returnForward) {
        throw std::runtime_error("ensemble-resident evaluation does not yet return full forward trajectories.");
    }
}

// Complains unless a direct call carries all of its arguments
void grumbleDirectArity(int nrhs) {
    if (nrhs < 8) {
        throw std::runtime_error("direct CUDA call expects drifts, controls, waveform, rho_init, rho_targ, dt, fidelity_type, integrator.");
    }
}

// Complains unless the integrator is one this backend implements
void grumbleIntegrator(const std::string& integrator) {
    if (integrator != "rectangle") {
        throw std::runtime_error("CUDA backend currently supports only rectangle integrator.");
    }
}

// Complains unless a command was given the arguments and outputs it needs
void grumbleCommandArity(bool argsOk, const char* expects, bool outputsOk, const char* returns) {
    if (!argsOk) throw std::runtime_error(std::string("this command expects: ") + expects + ".");
    if (!outputsOk) throw std::runtime_error(std::string("this command returns ") + returns + ".");
}

// Reports whether the first argument is a given command word
bool isCommand(const mxArray* arg, const char* expected) {
    if (!mxIsChar(arg)) return false;
    char buffer[128];
    if (mxGetString(arg, buffer, sizeof(buffer)) != 0) return false;
    return std::strcmp(buffer, expected) == 0;
}

// Reads a MATLAB character array as a C++ string
std::string getString(const mxArray* arg, const char* name) {
    grumbleString(arg, name);
    char* raw = mxArrayToString(arg);
    if (raw == nullptr) {
        throw std::runtime_error(std::string("could not read ") + name + ".");
    }
    std::string out(raw);
    mxFree(raw);
    return out;
}

// Reads an operator pack handle
std::uint64_t getHandle(const mxArray* arg) {
    grumbleHandle(arg);
    return static_cast<std::uint64_t*>(mxGetData(arg))[0];
}

// Returns the pack behind a handle, complaining if it is not a live one
OperatorPack* requirePack(std::uint64_t handle) {
    auto it = g_packs.find(handle);
    if (it == g_packs.end() || it->second == nullptr) {
        throw std::runtime_error("unknown or destroyed GPU operator-pack handle.");
    }
    return it->second;
}

}  // namespace

#endif  // GRAPE_GPU_GRUMBLE_CUH
