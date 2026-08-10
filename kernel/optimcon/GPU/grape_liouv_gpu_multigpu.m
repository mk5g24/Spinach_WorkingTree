% Multi-GPU ensemble GRAPE objective. Splits an ensemble of drift
% Liouvillians across MATLAB worker processes, one GPU per worker, and
% returns the ensemble-averaged fidelity and gradient. Each worker owns a
% process-local grape_liouv_gpu_mex operator pack and only ever sees its
% own sub-ensemble; there is no device to device communication, and the
% spmd barrier is the only synchronisation point. The client combines the
% local results weighted by sub-ensemble size. The command interface
% mirrors that of the MEX file itself. Syntax:
%
%      ctx=grape_liouv_gpu_multigpu('pack_ensemble',ensemble_drifts,...
%                                   controls,dt,gpu_devices)
%
%      [traj_data,fidelity,...
%       gradient]=grape_liouv_gpu_multigpu('eval',ctx,waveform,...
%                                          rho_init,rho_targ,...
%                                          fidelity_type,...
%                                          gradient_mode,return_forward)
%
%      grape_liouv_gpu_multigpu('destroy',ctx,clear_mex)
%
%      grape_liouv_gpu_multigpu('clear')
%
% Parameters:
%
%   ensemble_drifts - cell array with one drift Liouvillian, or
%                     one cell array of drift Liouvillians, per
%                     ensemble member
%
%   controls        - control operators in Liouville space (cell
%                     array of matrices), common to all members
%
%   dt              - time slice durations, seconds
%
%   gpu_devices     - row vector of GPU indices, one per worker
%
%   ctx             - context structure returned by pack_ensemble
%
%   waveform        - control coefficients for each control ope-
%                     rator (in vertical dimension) at each time
%                     slice (horizontal dimension), rad/s
%
%   rho_init        - initial state as a vector in Liouville space
%
%   rho_targ        - target state as a vector in Liouville space
%
%   fidelity_type   - 'real', 'imag', or 'square'
%
%   gradient_mode   - 'exact', 'exact_implicit', or 'midpoint'
%
%   return_forward  - must be false, see the note below
%
%   clear_mex       - true to unload the MEX file on the workers
%
% Outputs:
%
%   traj_data       - performance counters summed over the workers,
%                     with an empty forward trajectory field
%
%   fidelity        - ensemble-averaged fidelity
%
%   gradient        - ensemble-averaged gradient
%
% Note: per-member forward trajectories are not returned. Collecting
%       them would move the whole ensemble trajectory stack across the
%       worker boundary on every objective evaluation, which costs more
%       than the split saves.
%
% Note: putting more than one worker on the same device is permitted
%       and is slower than using that device from one worker. It exists
%       so that the cost of the split itself can be measured.
%
% m.keitel@soton.ac.uk

function varargout = grape_liouv_gpu_multigpu(command, varargin)

if nargin < 1 || ~ischar(command)
    error('first argument must be a command string.');
end

switch lower(command)
    case 'pack_ensemble'
        if nargin ~= 5
            error('pack_ensemble expects: ensemble_drifts, controls, dt, gpu_devices.');
        end
        varargout{1} = pack_ensemble(varargin{1}, varargin{2}, varargin{3}, varargin{4});

    case 'eval'
        if nargin < 6 || nargin > 8
            error('eval expects: ctx, waveform, rho_init, rho_targ, fidelity_type [, gradient_mode] [, return_forward].');
        end
        gradient_mode = 'exact_implicit';
        if nargin >= 7 && ~isempty(varargin{6})
            gradient_mode = varargin{6};
        end
        return_forward = false;
        if nargin >= 8 && ~isempty(varargin{7})
            return_forward = varargin{7};
        end
        [varargout{1:nargout}] = eval_ensemble(varargin{1}, varargin{2}, varargin{3}, ...
            varargin{4}, varargin{5}, gradient_mode, return_forward, nargout >= 3);

    case 'destroy'
        if nargin < 2 || nargin > 3
            error('destroy expects: ctx [, clear_mex].');
        end
        clear_mex = true;
        if nargin >= 3 && ~isempty(varargin{2})
            clear_mex = varargin{2};
        end
        destroy_context(varargin{1}, clear_mex);

    case 'clear'
        clear_worker_mex();

    otherwise
        error('unknown grape_liouv_gpu_multigpu command: %s', command);
end

end

% Splits the ensemble across workers and builds one GPU pack per worker
function ctx = pack_ensemble(ensemble_drifts, controls, dt, gpu_devices)
if ~iscell(ensemble_drifts) || isempty(ensemble_drifts)
    error('ensemble_drifts must be a non-empty cell array.');
end
gpu_devices = validate_gpu_devices(gpu_devices, numel(ensemble_drifts));
nworkers = numel(gpu_devices);
ensure_pool(nworkers);

this_dir = fileparts(mfilename('fullpath'));
controls = sparsify_ops_gpu(controls);
parts = balanced_partitions(numel(ensemble_drifts), nworkers);
members = cellfun(@numel, parts);

worker_dir = this_dir;
worker_devices = gpu_devices;
worker_parts = parts;
worker_controls = controls;
worker_dt = dt;

try
    spmd (nworkers)
        addpath(worker_dir);
        gpuDevice(worker_devices(spmdIndex));
        if exist('grape_liouv_gpu_mex', 'file') ~= 3
            error('grape_liouv_gpu_mex is not compiled on worker %d.', spmdIndex);
        end
        grape_liouv_gpu_mex('clear');
        local_drifts = ensemble_drifts(worker_parts{spmdIndex});
        local_handle = grape_liouv_gpu_mex('pack_ensemble', local_drifts, worker_controls, worker_dt);
        worker_ctx = struct('handle', local_handle, 'device', worker_devices(spmdIndex), ...
                            'members', numel(local_drifts));
    end
catch ME
    try
        clear_worker_mex();
    catch
    end
    rethrow(ME)
end

ctx = struct();
ctx.worker_ctx = worker_ctx;
ctx.nworkers = nworkers;
ctx.gpu_devices = gpu_devices(:).';
ctx.members = members(:).';
ctx.total_members = sum(members);
ctx.weights = members(:).' / sum(members);
ctx.partitions = parts;
ctx.directory = this_dir;
end

% Evaluates the objective on every worker and combines the local results
function [traj_data, fidelity, gradient] = eval_ensemble(ctx, waveform, rho_init, rho_targ, fidelity_type, gradient_mode, return_forward, need_gradient)
validate_context(ctx);
if return_forward
    error('multi-GPU ensemble wrapper does not return full per-member forward trajectories.');
end
if ~ismember(gradient_mode, {'exact', 'frechet', 'exact_implicit', 'frechet_implicit', 'midpoint'})
    error('unsupported gradient_mode: %s', gradient_mode);
end

worker_ctx = ctx.worker_ctx;
nworkers = ctx.nworkers;
wall = tic;
if need_gradient
    spmd (nworkers)
        local_ctx = worker_ctx;
        [traj_local, fid_local, grad_local] = grape_liouv_gpu_mex('eval', local_ctx.handle, ...
            waveform, rho_init, rho_targ, fidelity_type, gradient_mode, false);
    end
else
    spmd (nworkers)
        local_ctx = worker_ctx;
        [traj_local, fid_local] = grape_liouv_gpu_mex('eval', local_ctx.handle, ...
            waveform, rho_init, rho_targ, fidelity_type, gradient_mode, false);
        grad_local = [];
    end
end
eval_seconds = toc(wall);

fidelity = 0;
for n = 1:nworkers
    fidelity = fidelity + ctx.weights(n) * fid_local{n};
end

if need_gradient
    gradient = zeros(size(grad_local{1}));
    for n = 1:nworkers
        gradient = gradient + ctx.weights(n) * grad_local{n};
    end
end

traj_data = merge_traj_data(traj_local, ctx, eval_seconds);
end

% Destroys the per-worker packs and optionally unloads the MEX file
function destroy_context(ctx, clear_mex)
if isempty(ctx)
    return
end
validate_context(ctx);
worker_ctx = ctx.worker_ctx;
nworkers = ctx.nworkers;
spmd (nworkers)
    local_ctx = worker_ctx;
    if isstruct(local_ctx) && isfield(local_ctx, 'handle') && ~isempty(local_ctx.handle)
        try
            grape_liouv_gpu_mex('destroy', local_ctx.handle);
        catch
        end
    end
    if clear_mex
        try
            grape_liouv_gpu_mex('clear');
        catch
        end
    end
end
end

% Clears every pack held by the MEX file on all pool workers
function clear_worker_mex()
pool = gcp('nocreate');
if isempty(pool)
    return
end
nworkers = pool.NumWorkers;
spmd (nworkers)
    if exist('grape_liouv_gpu_mex', 'file') == 3
        try
            grape_liouv_gpu_mex('clear');
        catch
        end
    end
end
end

% Rejects a context that did not come out of pack_ensemble
function validate_context(ctx)
if ~isstruct(ctx) || ~isfield(ctx, 'worker_ctx') || ~isfield(ctx, 'nworkers') || ...
   ~isfield(ctx, 'weights') || ~isfield(ctx, 'members')
    error('invalid multi-GPU context.');
end
if ctx.nworkers < 1 || numel(ctx.weights) ~= ctx.nworkers
    error('invalid multi-GPU worker count.');
end
end

% Checks the device list against the backend and the ensemble size
function gpu_devices = validate_gpu_devices(gpu_devices, nensemble)
if nargin < 1 || isempty(gpu_devices)
    error('gpu_devices must list one GPU device id per MATLAB worker, for example [1 2].');
end
if ~isnumeric(gpu_devices) || ~isreal(gpu_devices) || any(gpu_devices(:) ~= fix(gpu_devices(:))) || any(gpu_devices(:) < 1)
    error('gpu_devices must contain positive integer GPU device ids.');
end
gpu_devices = double(gpu_devices(:)).';
if numel(gpu_devices) > nensemble
    error('gpu_devices has more workers than ensemble members.');
end
if exist('grape_liouv_gpu_mex', 'file') ~= 3
    error('grape_liouv_gpu_mex is not compiled; run compile_grape_gpu first.');
end
caps = grape_liouv_gpu_mex('capabilities');
if ~isfield(caps, 'cudaBackendReady') || ~caps.cudaBackendReady
    error('CUDA backend is not ready: %s', caps.message);
end
if ~isfield(caps, 'deviceCount') || any(gpu_devices > caps.deviceCount)
    error('requested GPU device exceeds available device count.');
end
end

% Starts a process pool of the required size, or checks the existing one
function ensure_pool(nworkers)
pool = gcp('nocreate');
if isempty(pool)
    parpool('Processes', nworkers);
elseif pool.NumWorkers < nworkers
    error('active parallel pool has %d workers, but %d are required.', pool.NumWorkers, nworkers);
end
end

% Splits a range of indices into nearly equal contiguous blocks
function parts = balanced_partitions(nitems, nparts)
edges = floor((0:nparts) * nitems / nparts);
parts = cell(1, nparts);
for n = 1:nparts
    parts{n} = (edges(n)+1):edges(n+1);
end
end

% Adds up the per-worker performance counters into one summary
function traj = merge_traj_data(traj_parts, ctx, eval_seconds)
traj = struct('forward', []);
traj.worker_devices = ctx.gpu_devices;
traj.worker_members = ctx.members;
traj.wall_seconds = eval_seconds;

fields = {'estimated_flops', 'estimated_bytes', 'estimated_sparse_nnz_visits', ...
          'affine_assembly_values', 'affine_assembly_calls', ...
          'sparse_kernel_calls', 'reduction_kernel_calls'};
perf = struct();
for f = 1:numel(fields)
    perf.(fields{f}) = 0;
end
have_perf = false;
for n = 1:ctx.nworkers
    local_traj = traj_parts{n};
    if isstruct(local_traj) && isfield(local_traj, 'performance') && isstruct(local_traj.performance)
        have_perf = true;
        for f = 1:numel(fields)
            if isfield(local_traj.performance, fields{f})
                value = local_traj.performance.(fields{f});
                if isnumeric(value) && isscalar(value)
                    perf.(fields{f}) = perf.(fields{f}) + double(value);
                end
            end
        end
    end
end
if have_perf
    perf.wall_seconds = eval_seconds;
    perf.worker_count = ctx.nworkers;
    perf.gpu_devices = ctx.gpu_devices;
    perf.ensemble_members = ctx.members;
    perf.gflops = perf.estimated_flops / max(eval_seconds, eps) / 1e9;
    perf.bandwidth_gbs = perf.estimated_bytes / max(eval_seconds, eps) / 1e9;
    perf.arithmetic_intensity = perf.estimated_flops / max(perf.estimated_bytes, eps);
    traj.performance = perf;
end
end

