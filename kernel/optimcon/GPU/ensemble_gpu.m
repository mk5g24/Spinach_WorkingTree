% CUDA-backed ensemble-averaged GRAPE objective. This is the GPU
% counterpart of Spinach ensemble.m, and it follows the same averaging
% conventions over drift ensembles, offset grids, power levels, state and
% target pairs, waveform distortions, and weighted control operator sets.
% A context holding one or more persistent GPU operator packs is created
% once and reused for every objective evaluation of an optimisation.
% Syntax:
%
%     context=ensemble_gpu('setup',spin_system,options)
%
%     [traj_data,fidelity,...
%      gradient]=ensemble_gpu(waveform,spin_system,context)
%
%     ensemble_gpu('cleanup',context,options)
%
% Parameters:
%
%   spin_system  - Spinach data object that has been through
%                  the optimcon.m problem setup function
%
%   waveform     - control coefficients for each control ope-
%                  rator (in vertical dimension) at each time
%                  slice (horizontal dimension), rad/s
%
%   context      - structure returned by the setup call, hol-
%                  ding the GPU operator pack handles and the
%                  ensemble bookkeeping
%
%   options      - option structure, see optimcon_gpu.m
%
% Outputs:
%
%   traj_data    - performance counters from the last ensemble
%                  member evaluated, with an empty forward tra-
%                  jectory field
%
%   fidelity     - ensemble-averaged fidelity
%
%   gradient     - ensemble-averaged gradient
%
% Note: calling the objective without a context creates a throwaway one
%       and destroys it afterwards, which re-uploads every operator on
%       every call. That is only appropriate for a single evaluation.
%
% Note: phase cycles, keyholes, prefix, suffix, and dead time hooks,
%       trajectory plotting, Hessians, and ensemble correlations are
%       rejected with an error. Falling back to different averaging
%       semantics without telling the caller would be worse.
%
% m.keitel@soton.ac.uk

function varargout = ensemble_gpu(varargin)

if ischar(varargin{1}) || (isstring(varargin{1}) && isscalar(varargin{1}))
    [varargout{1:nargout}] = command_dispatch(varargin{:});
    return
end

waveform = varargin{1};
spin_system = varargin{2};
if numel(varargin) >= 3
    context = varargin{3};
else
    options = struct();
    options.gradient_mode = 'exact_implicit';
    options.clear_mex_on_exit = true;
    context = setup_context(spin_system, options);
    cleanup = onCleanup(@() cleanup_context(context, options));
end

grumbles(spin_system, waveform, context);

n_outputs = nargout;
if n_outputs > 3
    error('ensemble_gpu:NoCudaHessian', 'CUDA GRAPE Hessians are not implemented.');
end

n_states = numel(spin_system.control.rho_init);
n_powers = numel(spin_system.control.pwr_levels);
n_distortions = size(spin_system.control.distortion, 1);
n_cases = n_states * n_powers * n_distortions;

fidelity_sum = 0;
if n_outputs > 2
    gradient_sum = zeros(size(waveform));
end
last_traj = struct('forward', []);

for s = 1:n_states
    rho_init = spin_system.control.rho_init{s};
    rho_targ = spin_system.control.rho_targ{s};
    for p = 1:n_powers
        power = spin_system.control.pwr_levels(p);
        for d = 1:n_distortions
            local_waveform = power * waveform;
            if n_outputs > 2
                jacobian = speye(numel(local_waveform));
                for k = 1:size(spin_system.control.distortion, 2)
                    distortion = spin_system.control.distortion{d,k};
                    [local_waveform, stage_jacobian] = distortion(local_waveform);
                    jacobian = stage_jacobian * jacobian;
                end
                [last_traj, fidelity, gradient] = eval_operator_ensemble(context, ...
                    local_waveform, rho_init, rho_targ, spin_system.control.fidelity, true);
                if ~isempty(spin_system.control.freeze)
                    gradient(spin_system.control.freeze) = 0;
                end
                gradient = jacobian' * gradient(:);
                gradient_sum = gradient_sum + power * reshape(gradient, size(waveform));
            else
                for k = 1:size(spin_system.control.distortion, 2)
                    distortion = spin_system.control.distortion{d,k};
                    local_waveform = distortion(local_waveform);
                end
                [last_traj, fidelity] = eval_operator_ensemble(context, ...
                    local_waveform, rho_init, rho_targ, spin_system.control.fidelity, false);
            end
            fidelity_sum = fidelity_sum + fidelity;
        end
    end
end

traj_data = {struct('forward', [])};
if isstruct(last_traj) && isfield(last_traj, 'performance')
    traj_data{1}.performance = last_traj.performance;
end

varargout = cell(1, n_outputs);
if n_outputs >= 1
    varargout{1} = traj_data;
end
if n_outputs >= 2
    varargout{2} = fidelity_sum / n_cases;
end
if n_outputs >= 3
    varargout{3} = gradient_sum / n_cases;
end

if exist('cleanup', 'var')
    clear cleanup
end

end

% Routes the setup and cleanup commands away from the objective call
function varargout = command_dispatch(command, varargin)
command = char(command);
switch command
    case 'setup'
        spin_system = varargin{1};
        if numel(varargin) >= 2 && ~isempty(varargin{2})
            options = varargin{2};
        else
            options = struct();
        end
        options = complete_options(options);
        varargout{1} = setup_context(spin_system, options);
    case 'cleanup'
        context = varargin{1};
        if numel(varargin) >= 2 && ~isempty(varargin{2})
            options = varargin{2};
        else
            options = struct();
        end
        options = complete_options(options);
        cleanup_context(context, options);
        varargout = {};
    otherwise
        error('unknown ensemble_gpu command: %s', command);
end
end

% Fills in the option fields that the caller did not set
function options = complete_options(options)
if ~isfield(options, 'gradient_mode') || isempty(options.gradient_mode)
    options.gradient_mode = 'exact_implicit';
end
if ~isfield(options, 'clear_mex_on_exit') || isempty(options.clear_mex_on_exit)
    options.clear_mex_on_exit = true;
end
if ~isfield(options, 'max_ensemble_chunk_size') || isempty(options.max_ensemble_chunk_size)
    options.max_ensemble_chunk_size = Inf;
end
end

% Builds the GPU operator packs and the ensemble bookkeeping
function context = setup_context(spin_system, options)
validate_supported_problem(spin_system, options);
ensemble_drifts = ensemble_drifts_gpu(spin_system);
drift_chunks = split_ensemble_drifts(ensemble_drifts, options.max_ensemble_chunk_size);
operator_ensembles = get_operator_ensembles(spin_system);
operator_weights = get_operator_weights(spin_system, numel(operator_ensembles));

handles = cell(numel(operator_ensembles), numel(drift_chunks));
try
    for n = 1:numel(operator_ensembles)
        controls = sparsify_ops_gpu(operator_ensembles{n});
        for k = 1:numel(drift_chunks)
            handles{n,k} = grape_liouv_gpu_mex('pack_ensemble', drift_chunks{k}, ...
                controls, spin_system.control.pulse_dt);
        end
    end
catch ME
    destroy_handles(handles);
    rethrow(ME)
end

context = struct();
context.handles = handles;
context.handle = handles{1,1};
context.gradient_mode = options.gradient_mode;
context.ensemble_members = numel(ensemble_drifts);
context.drift_chunk_sizes = cellfun(@numel, drift_chunks);
context.operator_members = numel(operator_ensembles);
context.operator_weights = operator_weights;
end

% Averages fidelity and gradient over the control operator ensemble
function [last_traj, fidelity, gradient] = eval_operator_ensemble(context, local_waveform, ...
                                                                  rho_init, rho_targ, fidelity_type, ...
                                                                  need_gradient)
if isfield(context, 'handles')
    handles = context.handles;
    operator_weights = context.operator_weights;
    if isfield(context, 'drift_chunk_sizes')
        chunk_weights = context.drift_chunk_sizes / sum(context.drift_chunk_sizes);
    else
        handles = reshape(handles, [], 1);
        chunk_weights = 1;
    end
else
    handles = {context.handle};
    operator_weights = 1;
    chunk_weights = 1;
end
fidelity = 0;
if need_gradient
    gradient = zeros(size(local_waveform));
end
last_traj = struct('forward', []);
for n = 1:size(handles, 1)
    local_fidelity_sum = 0;
    if need_gradient
        local_gradient_sum = zeros(size(local_waveform));
    end
    for k = 1:size(handles, 2)
        if need_gradient
            [last_traj, local_fidelity, local_gradient] = grape_liouv_gpu_mex('eval', ...
                handles{n,k}, local_waveform, rho_init, rho_targ, fidelity_type, ...
                context.gradient_mode, false);
            local_gradient_sum = local_gradient_sum + chunk_weights(k) * local_gradient;
        else
            [last_traj, local_fidelity] = grape_liouv_gpu_mex('eval', ...
                handles{n,k}, local_waveform, rho_init, rho_targ, fidelity_type, ...
                context.gradient_mode, false);
        end
        local_fidelity_sum = local_fidelity_sum + chunk_weights(k) * local_fidelity;
    end
    fidelity = fidelity + operator_weights(n) * local_fidelity_sum;
    if need_gradient
        gradient = gradient + operator_weights(n) * local_gradient_sum;
    end
end
end

% Cuts the drift ensemble into chunks that fit on the device
function drift_chunks = split_ensemble_drifts(ensemble_drifts, max_chunk_size)
if isinf(max_chunk_size) || (max_chunk_size >= numel(ensemble_drifts))
    drift_chunks = {ensemble_drifts};
    return
end
chunk_edges = 1:max_chunk_size:numel(ensemble_drifts);
drift_chunks = cell(1, numel(chunk_edges));
for n = 1:numel(chunk_edges)
    last = min(chunk_edges(n)+max_chunk_size-1, numel(ensemble_drifts));
    drift_chunks{n} = ensemble_drifts(chunk_edges(n):last);
end
end

% Returns the control operator sets, one per ensemble member
function operator_ensembles = get_operator_ensembles(spin_system)
if isfield(spin_system.control, 'operator_ensembles')
    operator_ensembles = reshape(spin_system.control.operator_ensembles, 1, []);
else
    operator_ensembles = {spin_system.control.operators};
end
ncontrols = numel(spin_system.control.operators);
for n = 1:numel(operator_ensembles)
    if (~iscell(operator_ensembles{n})) || (numel(operator_ensembles{n}) ~= ncontrols) || ...
       (~all(cellfun(@ismatrix, operator_ensembles{n}(:))))
        error('spin_system.control.operator_ensembles must contain equally sized cell arrays of matrices.');
    end
end
end

% Returns normalised weights for the control operator ensemble
function operator_weights = get_operator_weights(spin_system, noperators)
if isfield(spin_system.control, 'operator_weights')
    operator_weights = reshape(spin_system.control.operator_weights, 1, []);
    if (~isnumeric(operator_weights)) || (~isreal(operator_weights)) || ...
       (numel(operator_weights) ~= noperators) || any(~isfinite(operator_weights(:))) || ...
       any(operator_weights(:) < 0) || (sum(operator_weights(:)) <= 0)
        error('spin_system.control.operator_weights must be a finite non-negative vector matching the operator ensemble.');
    end
    operator_weights = operator_weights / sum(operator_weights);
else
    if noperators ~= 1
        error('spin_system.control.operator_weights is required for operator ensembles.');
    end
    operator_weights = 1;
end
end

% Releases every GPU operator pack held by a context
function destroy_handles(handles)
for n = 1:numel(handles)
    if ~isempty(handles{n})
        try
            grape_liouv_gpu_mex('destroy', handles{n});
        catch
        end
    end
end
end

% Rejects problem features the CUDA backend does not implement
function validate_supported_problem(spin_system, options)
if ~isstruct(spin_system) || ~isfield(spin_system, 'control')
    error('spin_system must be a Spinach object processed by optimcon().');
end
if ~isfield(spin_system, 'bas') || ~isfield(spin_system.bas, 'formalism') || ...
   ~ismember(spin_system.bas.formalism, {'sphten-liouv', 'zeeman-liouv'})
    error('ensemble_gpu currently supports only sphten-liouv and zeeman-liouv formalisms.');
end
if ~strcmp(spin_system.control.integrator, 'rectangle')
    error('ensemble_gpu currently supports only the rectangle integrator.');
end
if spin_system.control.dead_time ~= 0
    error('nonzero dead_time is not supported by ensemble_gpu.');
end
if ~isempty(spin_system.control.prefix) || ~isempty(spin_system.control.suffix)
    error('prefix/suffix sequence functions are not supported by ensemble_gpu.');
end
if any(~cellfun(@isempty, spin_system.control.keyholes(:)))
    error('keyhole operators are not supported by ensemble_gpu.');
end
if ~isempty(spin_system.control.plotting)
    error('disable control.plotting for ensemble_gpu.');
end
if ~isempty(spin_system.control.traj_opts)
    error('control.traj_opts must be empty for ensemble_gpu.');
end
if ~isempty(spin_system.control.phase_cycle)
    error('phase cycles are not supported by ensemble_gpu.');
end
if ~isempty(spin_system.control.ens_corrs)
    error('ensemble correlations are not supported by ensemble_gpu.');
end
if ~ismember(options.gradient_mode, {'exact', 'frechet', 'exact_implicit', 'frechet_implicit'})
    error('unsupported gradient_mode: %s', options.gradient_mode);
end
if (~isnumeric(options.max_ensemble_chunk_size)) || (~isreal(options.max_ensemble_chunk_size)) || ...
   (~isscalar(options.max_ensemble_chunk_size)) || ...
   (options.max_ensemble_chunk_size < 1) || ...
   ((~isinf(options.max_ensemble_chunk_size)) && ...
    ((~isfinite(options.max_ensemble_chunk_size)) || ...
     (fix(options.max_ensemble_chunk_size) ~= options.max_ensemble_chunk_size)))
    error('options.max_ensemble_chunk_size must be a positive integer or Inf.');
end
if exist('grape_liouv_gpu_mex', 'file') ~= 3
    error('grape_liouv_gpu_mex is not compiled; run compile_grape_gpu first.');
end
caps = grape_liouv_gpu_mex('capabilities');
if ~isfield(caps, 'cudaBackendReady') || ~caps.cudaBackendReady
    error('CUDA backend is not ready: %s', caps.message);
end
if ~isfield(caps, 'supportsEnsembleResidentPack') || ~caps.supportsEnsembleResidentPack
    error('CUDA backend does not report supportsEnsembleResidentPack=true.');
end
if ismember(options.gradient_mode, {'exact_implicit', 'frechet_implicit'}) && ...
   (~isfield(caps, 'supportsImplicitExactFrechet') || ~caps.supportsImplicitExactFrechet)
    error('CUDA backend does not report supportsImplicitExactFrechet=true.');
end
end

% Releases the packs and optionally unloads the MEX file
function cleanup_context(context, options)
if isstruct(context)
    if isfield(context, 'handles') && iscell(context.handles)
        destroy_handles(context.handles);
    elseif isfield(context, 'handle') && ~isempty(context.handle)
        try
            grape_liouv_gpu_mex('destroy', context.handle);
        catch
        end
    end
end
if options.clear_mex_on_exit && exist('grape_liouv_gpu_mex', 'file') == 3
    try
        grape_liouv_gpu_mex('clear');
    catch
    end
end
end

% Consistency enforcement
function grumbles(spin_system, waveform, context)
if ~isstruct(context)
    error('context must be created by ensemble_gpu(''setup'', spin_system, options).');
end
if isfield(context, 'handles')
    if (~iscell(context.handles)) || isempty(context.handles) || any(cellfun(@isempty, context.handles(:)))
        error('context must be created by ensemble_gpu(''setup'', spin_system, options).');
    end
elseif (~isfield(context, 'handle')) || isempty(context.handle)
    error('context must be created by ensemble_gpu(''setup'', spin_system, options).');
end
if ~isfield(spin_system, 'control')
    error('control data missing from spin_system, run optimcon() first.');
end
if (~isnumeric(waveform)) || (~isreal(waveform))
    error('waveform must be an array of real numbers.');
end
if size(waveform, 1) ~= numel(spin_system.control.operators)
    error('the number of rows in waveform must be equal to the number of controls.');
end
if size(waveform, 2) ~= spin_system.control.pulse_ntpts
    error('the number of columns in waveform must be equal to the number of time points.');
end
end

