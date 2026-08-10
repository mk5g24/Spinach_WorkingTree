% Runs the standard Spinach fmaxnewton.m optimiser against a CUDA-backed
% Cartesian GRAPE objective that holds one persistent ensemble operator
% pack on the GPU for the whole optimisation. No Spinach source file is
% modified: the optimiser, line search, penalties, bounds, freeze masks,
% and checkpointing remain Spinach's own code. Use optimcon_gpu.m instead
% of this function unless the problem has already been through optimcon.m.
% Syntax:
%
%     [x,data]=fmaxnewton_gpu(spin_system,guess,options)
%
% Parameters:
%
%   spin_system  - Spinach data object that has been through
%                  the optimcon.m problem setup function
%
%   guess        - initial guess for the waveform, in the norm-
%                  alised Cartesian form expected by grape_xy.m
%                  and fmaxnewton.m, not in rad/s
%
%   options.gradient_mode
%                - 'exact_implicit' for the fused implicit
%                  Frechet recurrence, or 'exact' for the
%                  split-action Frechet derivatives
%
%   options.gpu_devices
%                - empty to use the currently selected device,
%                  or a row vector of device indices such as
%                  [1 2] to split the ensemble across workers
%
%   options.clear_mex_on_exit
%                - true to release all GPU operator packs when
%                  the optimisation finishes
%
% Outputs:
%
%   x            - optimised waveform, same shape as guess
%
%   data         - optimiser diagnostics from fmaxnewton.m
%
% Note: the supported problem class is deliberately narrow. Liouville
%       space state vector formalisms, the rectangle integrator, and
%       the LBFGS or RBFGS optimiser methods are accepted, together
%       with drift ensembles, power levels, and offset grids. Prefix,
%       suffix, dead time, keyholes, plotting, trajectory averaging,
%       phase cycles, waveform distortions, and multiple state-target
%       pairs are rejected with an error. CUDA Hessians do not exist.
%
% Note: the drift generators come from the parallel pool ValueStore
%       where optimcon.m put them, so a pool must be active.
%
% m.keitel@soton.ac.uk

function [x,data]=fmaxnewton_gpu(spin_system,guess,options)

% Fill in the option fields that the caller did not set
options = complete_options(options);

validate_gpu_problem(spin_system, guess, options);
context = build_gpu_context(spin_system, options);
cleanup = onCleanup(@() cleanup_gpu_context(context, options));

[x, data] = fmaxnewton(spin_system, @(waveform, ss) grape_xy_gpu_obj(waveform, ss, context), guess);

end

% Cartesian GRAPE objective that fmaxnewton.m calls on every iteration
function [traj_data, fidelity, grad, hess] = grape_xy_gpu_obj(waveform, spin_system, context)

% Hessians are not implemented on the device
if nargout > 3
    error('fmaxnewton_gpu:NoCudaHessian', 'CUDA GRAPE Hessians are not implemented; use LBFGS or RBFGS.');
end

% The grape_xy.m input checks that matter for this objective
if ~isfield(spin_system, 'control')
    error('control data missing from spin_system, run optimcon() first.');
end
if (~isnumeric(waveform)) || (~isreal(waveform))
    error('waveform must be an array of real numbers.');
end
if size(waveform, 1) ~= numel(spin_system.control.operators)
    error('the number of rows in waveform must be equal to the number of controls.');
end

npenterms = numel(spin_system.control.penalties);

% Waveform basis translation, exactly as in grape_xy.m
if ~isempty(spin_system.control.basis)
    expanded_waveform = waveform * spin_system.control.basis;
else
    expanded_waveform = waveform;
end

if nargout == 2
    fidelity = zeros(1, npenterms + 1);
    [traj_data, fidelity(1)] = gpu_simulation(expanded_waveform, spin_system, context, false);
    for n = 1:npenterms
        pen = penalty(expanded_waveform, spin_system.control.penalties{n}, ...
                      spin_system.control.l_bound, spin_system.control.u_bound);
        fidelity(n+1) = spin_system.control.p_weights(n) * pen;
    end

elseif nargout == 3
    fidelity = zeros(1, npenterms + 1);
    grad = zeros(size(expanded_waveform, 1), size(expanded_waveform, 2), npenterms + 1);
    [traj_data, fidelity(1), grad(:,:,1)] = gpu_simulation(expanded_waveform, spin_system, context, true);
    for n = 1:npenterms
        [pen, pen_grad] = penalty(expanded_waveform, spin_system.control.penalties{n}, ...
                                  spin_system.control.l_bound, spin_system.control.u_bound);
        fidelity(n+1) = spin_system.control.p_weights(n) * pen;
        grad(:,:,n+1) = spin_system.control.p_weights(n) * pen_grad;
    end

    % Gradients go back to waveform basis coordinates, as in grape_xy.m
    if ~isempty(spin_system.control.basis)
        grad = tensorprod(spin_system.control.basis, grad, 2, 2);
        grad = permute(grad, [2 1 3]);
    end
end

% The symbol has to exist for MATLAB nargout bookkeeping
hess = [];

end

% Averages fidelity and gradient over states, power levels, and offsets
function [traj_data, fidelity, gradient] = gpu_simulation(waveform, spin_system, context, need_gradient)
nstates = numel(spin_system.control.rho_init);
npowers = numel(spin_system.control.pwr_levels);
ntotal = nstates * npowers;
fidelity_sum = 0;
if need_gradient
    gradient_sum = zeros(size(waveform));
end
last_traj = struct('forward', []);

for s = 1:nstates
    rho_init = spin_system.control.rho_init{s};
    rho_targ = spin_system.control.rho_targ{s};
    for p = 1:npowers
        power = spin_system.control.pwr_levels(p);
        physical_waveform = power * waveform;
        if need_gradient
            if isfield(context, 'multi_gpu') && context.multi_gpu
                [last_traj, fid, grad_phys] = grape_liouv_gpu_multigpu('eval', context.multi_gpu_context, ...
                    physical_waveform, rho_init, rho_targ, spin_system.control.fidelity, ...
                    context.gradient_mode, false);
            else
                [last_traj, fid, grad_phys] = grape_liouv_gpu_mex('eval', context.handle, ...
                    physical_waveform, rho_init, rho_targ, spin_system.control.fidelity, ...
                    context.gradient_mode, false);
            end
            fidelity_sum = fidelity_sum + fid;
            gradient_sum = gradient_sum + power * grad_phys;
        else
            if isfield(context, 'multi_gpu') && context.multi_gpu
                [last_traj, fid] = grape_liouv_gpu_multigpu('eval', context.multi_gpu_context, ...
                    physical_waveform, rho_init, rho_targ, spin_system.control.fidelity, ...
                    context.gradient_mode, false);
            else
                [last_traj, fid] = grape_liouv_gpu_mex('eval', context.handle, ...
                    physical_waveform, rho_init, rho_targ, spin_system.control.fidelity, ...
                    context.gradient_mode, false);
            end
            fidelity_sum = fidelity_sum + fid;
        end
    end
end

fidelity = fidelity_sum / ntotal;
if need_gradient
    gradient = gradient_sum / ntotal;
end

% Plotting is rejected upstream, so fmaxnewton.m only stores this
traj_data = {struct('forward', [])};
if isstruct(last_traj) && isfield(last_traj, 'performance')
    traj_data{1}.performance = last_traj.performance;
end

end

% Fills in the option fields that the caller did not set
function options = complete_options(options)
if ~isfield(options, 'gradient_mode') || isempty(options.gradient_mode)
    options.gradient_mode = 'exact_implicit';
end
if ~isfield(options, 'gpu_devices') || isempty(options.gpu_devices)
    options.gpu_devices = [];
end
if ~isfield(options, 'clear_mex_on_exit') || isempty(options.clear_mex_on_exit)
    options.clear_mex_on_exit = true;
end
end

% Rejects problem features the CUDA backend does not implement
function validate_gpu_problem(spin_system, guess, options)
if ~isstruct(spin_system) || ~isfield(spin_system, 'control')
    error('spin_system must be a Spinach object processed by optimcon().');
end
if (~isnumeric(guess)) || (~isreal(guess))
    error('guess must be a real numeric waveform array.');
end
if size(guess, 1) ~= numel(spin_system.control.operators)
    error('guess row count must match the number of control operators.');
end
if ~isempty(spin_system.control.basis)
    if size(guess, 2) ~= size(spin_system.control.basis, 1)
        error('with control.basis, guess columns must match the number of basis functions.');
    end
else
    if size(guess, 2) ~= spin_system.control.pulse_ntpts
        error('guess columns must match spin_system.control.pulse_ntpts.');
    end
end
if ~isfield(spin_system, 'bas') || ~isfield(spin_system.bas, 'formalism') || ...
   ~ismember(spin_system.bas.formalism, {'sphten-liouv', 'zeeman-liouv'})
    error('fmaxnewton_gpu currently supports only sphten-liouv and zeeman-liouv formalisms.');
end
if ~strcmp(spin_system.control.integrator, 'rectangle')
    error('fmaxnewton_gpu currently supports only the rectangle integrator.');
end
if ~ismember(spin_system.control.method, {'lbfgs', 'rbfgs'})
    error('CUDA Hessians are not implemented; use control.method = ''lbfgs'' or ''rbfgs''.');
end
if spin_system.control.dead_time ~= 0
    error('nonzero dead_time is not supported by the CUDA ensemble fmaxnewton wrapper.');
end
if ~isempty(spin_system.control.prefix) || ~isempty(spin_system.control.suffix)
    error('prefix/suffix sequence functions are not supported by the CUDA ensemble fmaxnewton wrapper.');
end
if any(~cellfun(@isempty, spin_system.control.keyholes(:)))
    error('keyhole operators are not supported by the CUDA ensemble fmaxnewton wrapper.');
end
if ~isempty(spin_system.control.plotting)
    error('disable control.plotting for fmaxnewton_gpu; CUDA ensemble calls do not return plotting trajectories.');
end
if ~isempty(spin_system.control.traj_opts)
    error('control.traj_opts must be empty for fmaxnewton_gpu.');
end
if ~isempty(spin_system.control.phase_cycle)
    error('phase cycles are not supported by fmaxnewton_gpu.');
end
if ~is_no_distortion(spin_system.control.distortion)
    error('waveform distortions are not supported by fmaxnewton_gpu.');
end
if numel(spin_system.control.rho_init) ~= 1 || numel(spin_system.control.rho_targ) ~= 1
    error('fmaxnewton_gpu currently supports one state-target pair.');
end
if ~isempty(spin_system.control.ens_corrs)
    error('ensemble correlations are not supported by fmaxnewton_gpu.');
end
if ~isfield(spin_system.control, 'ndrifts') || spin_system.control.ndrifts < 1
    error('spin_system.control.ndrifts is missing; run optimcon() with control.drifts first.');
end
if ~ismember(options.gradient_mode, {'exact', 'frechet', 'exact_implicit', 'frechet_implicit'})
    error('unsupported gradient_mode: %s', options.gradient_mode);
end
if ~isempty(options.gpu_devices)
    if ~isnumeric(options.gpu_devices) || ~isreal(options.gpu_devices) || ...
       any(options.gpu_devices(:) ~= fix(options.gpu_devices(:))) || any(options.gpu_devices(:) < 1)
        error('options.gpu_devices must contain positive integer GPU device ids.');
    end
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

% Reports whether a distortion specification is the identity
function tf = is_no_distortion(distortion)
if isempty(distortion)
    tf = true;
    return
end
if ~iscell(distortion)
    tf = false;
    return
end
tf = true;
for n = 1:numel(distortion)
    if ~isa(distortion{n}, 'function_handle') || ~strcmp(func2str(distortion{n}), 'no_dist')
        tf = false;
        return
    end
end
end

% Packs the drift ensemble and controls onto the device
function context = build_gpu_context(spin_system, options)
ensemble_drifts = ensemble_drifts_gpu(spin_system);
controls = sparsify_ops_gpu(spin_system.control.operators);
context = struct();
context.gradient_mode = options.gradient_mode;
context.ensemble_members = numel(ensemble_drifts);
context.multi_gpu = false;
context.handle = [];
context.multi_gpu_context = [];

if numel(options.gpu_devices) > 1
    context.multi_gpu_context = grape_liouv_gpu_multigpu('pack_ensemble', ensemble_drifts, ...
        controls, spin_system.control.pulse_dt, options.gpu_devices);
    context.multi_gpu = true;
    return
elseif isscalar(options.gpu_devices)
    gpuDevice(options.gpu_devices);
end

handle = [];
try
    handle = grape_liouv_gpu_mex('pack_ensemble', ensemble_drifts, ...
        controls, spin_system.control.pulse_dt);
catch ME
    if ~isempty(handle)
        try
            grape_liouv_gpu_mex('destroy', handle);
        catch
        end
    end
    rethrow(ME)
end
context.handle = handle;
end

% Releases the packs and optionally unloads the MEX file
function cleanup_gpu_context(context, options)
if isstruct(context) && isfield(context, 'multi_gpu') && context.multi_gpu
    try
        grape_liouv_gpu_multigpu('destroy', context.multi_gpu_context, options.clear_mex_on_exit);
    catch
    end
elseif isstruct(context) && isfield(context, 'handle') && ~isempty(context.handle)
    try
        grape_liouv_gpu_mex('destroy', context.handle);
    catch
    end
end
if options.clear_mex_on_exit && exist('grape_liouv_gpu_mex', 'file') == 3
    try
        grape_liouv_gpu_mex('clear');
    catch
    end
end
end

