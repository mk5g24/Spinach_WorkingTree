% Top level entry point into GPU optimal control. Sets the problem up with
% the standard Spinach optimcon.m, and then runs the standard Spinach
% fmaxnewton.m optimiser against a CUDA-backed ensemble objective. No
% Spinach source file is modified: the optimiser, line search, penalties,
% bounds, freeze masks, and checkpointing all remain Spinach's own code,
% and only the objective function evaluation moves to the GPU. Syntax:
%
%     [pulse,data,...
%      spin_system]=optimcon_gpu(spin_system,control,guess,options)
%
% Parameters:
%
%   spin_system  - Spinach data object from create.m and basis.m
%
%   control      - optimal control problem specification, as
%                  taken by optimcon.m; control.operators may
%                  also be a cell array of cell arrays, which
%                  specifies a weighted ensemble of control
%                  operator sets, with the weights supplied in
%                  control.operator_weights and normalised here
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
%   options.clear_mex_on_exit
%                - true to release all GPU operator packs when
%                  the optimisation finishes
%
%   options.max_ensemble_chunk_size
%                - largest number of ensemble members held on
%                  the device at once, Inf for no chunking
%
% Outputs:
%
%   pulse        - optimised waveform, same shape as guess
%
%   data         - optimiser diagnostics from fmaxnewton.m
%
%   spin_system  - the spin system object after optimcon.m,
%                  with the control operator ensemble installed
%
% Note: the supported problem class is deliberately narrow. Liouville
%       space state vector formalisms, the rectangle integrator, and
%       the LBFGS or RBFGS optimiser methods are accepted; anything
%       else is rejected with an error rather than silently run with
%       different semantics. CUDA Hessians are not implemented.
%
% Note: a parallel pool is required because optimcon.m stores the drift
%       generators in the pool ValueStore. One is started if necessary.
%
% m.keitel@soton.ac.uk

function [pulse,data,spin_system]=optimcon_gpu(spin_system,control,guess,options)

% Fill in the option fields that the caller did not set
options = complete_options(options);
[control, operator_ensembles, operator_weights] = extract_operator_ensemble(control);

if isempty(gcp('nocreate'))
    parpool('Processes', 1);
end

spin_system = optimcon(spin_system, control);
spin_system = install_operator_ensemble(spin_system, operator_ensembles, operator_weights);
validate_gpu_problem(spin_system, guess, options);

context = ensemble_gpu('setup', spin_system, options);
cleanup = onCleanup(@() ensemble_gpu('cleanup', context, options));

[pulse, data] = fmaxnewton(spin_system, ...
    @(waveform, ss) grape_xy_gpu_obj(waveform, ss, context), guess);

end

% Cartesian GRAPE objective that fmaxnewton.m calls on every iteration
function [traj_data, fidelity, grad, hess] = grape_xy_gpu_obj(waveform, spin_system, context)

% Hessians are not implemented on the device
if nargout > 3
    error('optimcon_gpu:NoCudaHessian', 'CUDA GRAPE Hessians are not implemented; use LBFGS or RBFGS.');
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

npenterms = numel(spin_system.control.penalties);

if ~isempty(spin_system.control.basis)
    expanded_waveform = waveform * spin_system.control.basis;
else
    expanded_waveform = waveform;
end

if nargout == 2
    fidelity = zeros(1, npenterms + 1);
    [traj_data, fidelity(1)] = ensemble_gpu(expanded_waveform, spin_system, context);
    for n = 1:npenterms
        pen = penalty(expanded_waveform, spin_system.control.penalties{n}, ...
                      spin_system.control.l_bound, spin_system.control.u_bound);
        fidelity(n+1) = spin_system.control.p_weights(n) * pen;
    end

elseif nargout == 3
    fidelity = zeros(1, npenterms + 1);
    grad = zeros(size(expanded_waveform, 1), size(expanded_waveform, 2), npenterms + 1);
    [traj_data, fidelity(1), grad(:,:,1)] = ensemble_gpu(expanded_waveform, spin_system, context);
    for n = 1:npenterms
        [pen, pen_grad] = penalty(expanded_waveform, spin_system.control.penalties{n}, ...
                                  spin_system.control.l_bound, spin_system.control.u_bound);
        fidelity(n+1) = spin_system.control.p_weights(n) * pen;
        grad(:,:,n+1) = spin_system.control.p_weights(n) * pen_grad;
    end

    if ~isempty(spin_system.control.basis)
        grad = tensorprod(spin_system.control.basis, grad, 2, 2);
        grad = permute(grad, [2 1 3]);
    end
end

hess = [];

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
    error('optimcon_gpu currently supports only sphten-liouv and zeeman-liouv formalisms.');
end
if ~strcmp(spin_system.control.integrator, 'rectangle')
    error('optimcon_gpu currently supports only the rectangle integrator.');
end
if ~ismember(spin_system.control.method, {'lbfgs', 'rbfgs'})
    error('CUDA Hessians are not implemented; use control.method = ''lbfgs'' or ''rbfgs''.');
end
if spin_system.control.dead_time ~= 0
    error('nonzero dead_time is not supported by optimcon_gpu.');
end
if ~isempty(spin_system.control.prefix) || ~isempty(spin_system.control.suffix)
    error('prefix/suffix sequence functions are not supported by optimcon_gpu.');
end
if any(~cellfun(@isempty, spin_system.control.keyholes(:)))
    error('keyhole operators are not supported by optimcon_gpu.');
end
if ~isempty(spin_system.control.plotting)
    error('disable control.plotting for optimcon_gpu; CUDA ensemble calls do not return plotting trajectories.');
end
if ~isempty(spin_system.control.traj_opts)
    error('control.traj_opts must be empty for optimcon_gpu.');
end
if ~isempty(spin_system.control.phase_cycle)
    error('phase cycles are not supported by optimcon_gpu.');
end
if ~isempty(spin_system.control.ens_corrs)
    error('ensemble correlations are not supported by optimcon_gpu.');
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

% Separates a control operator ensemble from a plain operator set
function [control, operator_ensembles, operator_weights] = extract_operator_ensemble(control)
operator_ensembles = {};
operator_weights = [];
if isfield(control, 'operators') && iscell(control.operators) && ...
   ~isempty(control.operators) && all(cellfun(@iscell, control.operators(:)))
    if ~isfield(control, 'operator_weights')
        error('nested control.operators requires control.operator_weights.');
    end
    operator_ensembles = reshape(control.operators, 1, []);
    if (~isnumeric(control.operator_weights)) || (~isreal(control.operator_weights)) || ...
       (~isvector(control.operator_weights)) || ...
       (numel(control.operator_weights) ~= numel(operator_ensembles)) || ...
       any(~isfinite(control.operator_weights(:))) || any(control.operator_weights(:) < 0) || ...
       (sum(control.operator_weights(:)) <= 0)
        error('control.operator_weights must be a finite non-negative vector matching the operator ensemble.');
    end
    ncontrols = numel(operator_ensembles{1});
    for n = 1:numel(operator_ensembles)
        if (~iscell(operator_ensembles{n})) || (numel(operator_ensembles{n}) ~= ncontrols) || ...
           (~all(cellfun(@ismatrix, operator_ensembles{n}(:))))
            error('control.operators must contain equally sized cell arrays of matrices.');
        end
    end
    operator_weights = reshape(control.operator_weights, 1, []);
    operator_weights = operator_weights / sum(operator_weights);
    control.operators = operator_ensembles{1};
    control = rmfield(control, 'operator_weights');
elseif isfield(control, 'operator_weights')
    error('control.operator_weights may only be used with nested control.operators.');
end
end

% Puts the control operator ensemble into the spin system object
function spin_system = install_operator_ensemble(spin_system, operator_ensembles, operator_weights)
if isempty(operator_ensembles)
    return
end
spin_system.control.operator_ensembles = operator_ensembles;
for n = 1:numel(operator_ensembles)
    for k = 1:numel(operator_ensembles{n})
        spin_system.control.operator_ensembles{n}{k} = clean_up(spin_system, operator_ensembles{n}{k}, ...
            spin_system.tols.liouv_zero);
    end
end
spin_system.control.operator_ensembles{1} = spin_system.control.operators;
spin_system.control.operator_weights = operator_weights;
report(spin_system, [pad('Control operator ensemble size', 60) ...
                    int2str(numel(spin_system.control.operator_ensembles))]);
end

