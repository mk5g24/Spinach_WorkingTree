% Multi-GPU entry point into the GPU GRAPE optimiser. Splits the ensemble
% across MATLAB workers, one worker per GPU, and hands the problem to
% fmaxnewton_gpu.m with the device list installed in the options. The
% single-GPU path in fmaxnewton_gpu.m is unaffected and remains the
% default; this function is the only way to opt into more than one
% device. Syntax:
%
%      [x,data]=fmaxnewton_gpu_multigpu(spin_system,guess,...
%                                       gpu_devices,options)
%
% Parameters:
%
%   spin_system   - Spinach data object that has been through
%                   the optimcon.m problem setup function
%
%   guess         - initial guess for the waveform, a matrix
%                   with one row per control operator and one
%                   column per time slice
%
%   gpu_devices   - row vector of GPU indices, one per MATLAB
%                   worker, for example [1 2] to use two dev-
%                   ices; repeated indices are permitted and
%                   put several workers on the same device
%
%   options       - optimiser options structure, as taken by
%                   fmaxnewton_gpu.m; the gpu_devices field
%                   is overwritten by this function
%
% Outputs:
%
%   x             - optimised waveform, same shape as guess
%
%   data          - optimiser diagnostics from fmaxnewton.m
%
% Note: workers are synchronised by an spmd barrier and there is no
%       device to device communication. Each worker returns the
%       fidelity and gradient of its own sub-ensemble, and the client
%       combines them weighted by sub-ensemble size.
%
% Note: putting several workers on one device is slower than using
%       that device from a single worker, and is only useful as a
%       control experiment when benchmarking the split.
%
% m.keitel@soton.ac.uk

function [x,data]=fmaxnewton_gpu_multigpu(spin_system,guess,gpu_devices,options)

% Check consistency
grumble(gpu_devices,options);

% The device list is the only thing this wrapper adds
options.gpu_devices=gpu_devices;

% Everything else is the single-GPU optimiser
[x,data]=fmaxnewton_gpu(spin_system,guess,options);

end

% Consistency enforcement
function grumble(gpu_devices,options)
if (~isnumeric(gpu_devices))||(~isrow(gpu_devices))||isempty(gpu_devices)
    error('gpu_devices must be a non-empty row vector of GPU indices.');
end
if any(gpu_devices~=round(gpu_devices))||any(gpu_devices<1)
    error('gpu_devices must contain positive integers.');
end
if ~isstruct(options)
    error('options must be a structure.');
end
end

