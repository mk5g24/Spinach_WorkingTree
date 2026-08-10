/*
 * grape_liouv_gpu_mex.cpp
 *
 * Non-CUDA capability stub for the CUDA-accelerated Liouville-space GRAPE
 * backend.
 *
 * The real backend lives in grape_liouv_gpu_cuda_mex.cu and is built with
 * `make cuda-mex` on CUDA hosts.  This file is the default `make mex`
 * target. Machines without CUDA still get a safe MEX that refuses numerical
 * work.
 *
 * Build:
 *   mex -R2018a grape_liouv_gpu_mex.cpp -output grape_liouv_gpu_mex
 */

#include "mex.h"
#include <cstring>

namespace {

// Routine to check if supplied command does exsit
bool isCommand(const mxArray* arg, const char* expected) {
    if (!mxIsChar(arg) || mxGetNumberOfElements(arg) != std::strlen(expected)) {
        return false;
    }
    char buffer[64];
    if (std::strlen(expected) >= sizeof(buffer)) {
        return false;
    }
    if (mxGetString(arg, buffer, sizeof(buffer)) != 0) {
        return false;
    }
    return std::strcmp(buffer, expected) == 0;
}

// Fallback routine to create a dummy struction for devices with no CUDA support
mxArray* makeCapabilities() {
    const char* fields[] = {
        "name",
        "version",
        "cudaBackendReady",
        "supportsRectangleGradient",
        "supportsMidpointGradient",
        "supportsExactFrechetGradient",
        "supportsHessian",
        "supportsPersistentPack",
        "message"
    };
    mxArray* caps = mxCreateStructMatrix(1, 1, 9, fields);
    mxSetField(caps, 0, "name", mxCreateString("grape_liouv_gpu_mex_stub"));
    mxSetField(caps, 0, "version", mxCreateDoubleScalar(3.0));
    mxSetField(caps, 0, "cudaBackendReady", mxCreateLogicalScalar(false));
    mxSetField(caps, 0, "supportsRectangleGradient", mxCreateLogicalScalar(false));
    mxSetField(caps, 0, "supportsMidpointGradient", mxCreateLogicalScalar(false));
    mxSetField(caps, 0, "supportsExactFrechetGradient", mxCreateLogicalScalar(false));
    mxSetField(caps, 0, "supportsHessian", mxCreateLogicalScalar(false));
    mxSetField(caps, 0, "supportsPersistentPack", mxCreateLogicalScalar(false));
    mxSetField(caps, 0, "message", mxCreateString(
        "Non-CUDA safety stub. Build the real backend with `make cuda-mex` on a CUDA host."));
    return caps;
}

} // namespace

// Fallback mex routine that returns a dummy structure and warns about non CUDA device
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs == 1 && isCommand(prhs[0], "capabilities")) {
        if (nlhs > 1) {
            mexErrMsgIdAndTxt("GRAPE:LiouvGPU:TooManyOutputs",
                              "grape_liouv_gpu_mex('capabilities') returns one struct.");
        }
        plhs[0] = makeCapabilities();
        return;
    }

    if (nrhs == 1 && isCommand(prhs[0], "version")) {
        if (nlhs > 1) {
            mexErrMsgIdAndTxt("GRAPE:LiouvGPU:TooManyOutputs",
                              "grape_liouv_gpu_mex('version') returns one scalar.");
        }
        plhs[0] = mxCreateDoubleScalar(3.0);
        return;
    }

    mexErrMsgIdAndTxt("GRAPE:LiouvGPU:CudaBackendNotReady",
        "This is the non-CUDA safety stub. Build grape_liouv_gpu_cuda_mex.cu with `make cuda-mex` "
        "on a CUDA host to enable persistent operator packs and GPU GRAPE evaluation.");
}
