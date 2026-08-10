% Gradient Ascent Pulse Engineering (GRAPE) objective function and gradient
% on a GPU. This function is a drop-in replacement for grape_liouv.m: it is
% called with the same arguments, returns the same quantities, and picks the
% fastest backend that is known to reproduce grape_liouv.m exactly for the
% problem in hand. Three backends are tried in order: the CUDA/MEX backend
% in grape_liouv_gpu_cuda_mex.cu, a MATLAB gpuArray reference translation of
% grape_liouv.m, and finally grape_liouv.m itself. Syntax:
%
%        [traj_data,fidelity,...
%         grad]=grape_liouv_gpu(spin_system,drifts,controls,...
%                               waveform,rho_init,rho_targ,...
%                               fidelity_type)
%
% Parameters:
%
%   spin_system         - Spinach data object that has been through
%                         the optimcon.m problem setup function
%
%   drifts              - the drift Liouvillians: a cell array con-
%                         taining one matrix (for time-independent
%                         drift) or multiple matrices (one per time
%                         slice / point, for time-dependent drift)
%
%   controls            - control operators in Liouville space (cell
%                         array of matrices)
%
%   waveform            - control coefficients for each control ope-
%                         rator (in vertical dimension) at each time
%                         slice / point (horizontal dimension), rad/s
%
%   rho_init            - initial state of the system as a vector in
%                         Liouville space
%
%   rho_targ            - target state of the system as a vector in
%                         Liouville space
%
%   fidelity_type       - 'real'   (real part of the overlap)
%                         'imag'   (imaginary part of the overlap)
%                         'square' (absolute square of the overlap)
%
% Outputs:
%
%   traj_data.forward   - forward trajectory from the initial condi-
%                         tion, returned empty unless the plotting
%                         settings in spin_system.control need it
%
%   fidelity            - fidelity of the control sequence
%
%   grad                - gradient of the fidelity with respect to
%                         the control sequence
%
% Note: the number of requested outputs is honoured, work for outputs
%       that were not asked for is not done, and asking for a Hessian
%       sends the whole problem to grape_liouv.m.
%
% Note: the CUDA backend is only used when grape_liouv_gpu_mex reports
%       cudaBackendReady=true from its capability query. An incomplete
%       or stale MEX file that silently returns zeros is far worse than
%       a slow one, so the flag is checked on every call rather than
%       assumed from the presence of the binary.
%
% Note: the CUDA operator pack is cached between calls and keyed on the
%       drift and control operators and the time grid. The cache key is
%       compared with isequal, which costs a host-side comparison but
%       avoids a device re-upload on every objective evaluation. Call
%       grape_liouv_gpu_mex('pack',...) and ('eval',...) directly to
%       bypass that comparison.
%
% m.keitel@soton.ac.uk

function varargout=grape_liouv_gpu(spin_system,drifts,controls,waveform,...
                                   rho_init,rho_targ,fidelity_type)

% Check consistency
grumble(spin_system,drifts,controls,waveform,rho_init,rho_targ);

% Callers that ask for nothing still expect the side effects of the reference
n_outputs=nargout;
if n_outputs==0
    grape_liouv(spin_system,drifts,controls,waveform,rho_init,rho_targ,fidelity_type);
    return
end

% Fastest backend first, with a persistent GPU operator pack across calls
if mex_backend_ready()&&rectangle_no_hooks(spin_system,n_outputs)
    [varargout{1:n_outputs}]=cuda_backend(spin_system,drifts,controls,...
                                          waveform,rho_init,rho_targ,...
                                          fidelity_type,n_outputs);
    return
end

% MATLAB gpuArray translation of grape_liouv.m when a device is present
if matlab_gpu_supported(spin_system,n_outputs)
    [varargout{1:n_outputs}]=gpuarray_backend(spin_system,drifts,controls,...
                                              waveform,rho_init,rho_targ,...
                                              fidelity_type,n_outputs);
    return
end

% Exact CPU reference for every unsupported or non-GPU environment
[varargout{1:n_outputs}]=grape_liouv(spin_system,drifts,controls,...
                                     waveform,rho_init,rho_targ,fidelity_type);

end

% Reports whether a MEX backend exists and declares itself complete
function ok=mex_backend_ready()
ok=false;
if isempty(which('grape_liouv_gpu_mex'))
    return
end
try
    caps=grape_liouv_gpu_mex('capabilities');
    ok=isstruct(caps)&&isfield(caps,'cudaBackendReady')&&...
       islogical(caps.cudaBackendReady)&&caps.cudaBackendReady;
catch
    ok=false;
end
end

% Reports whether a CUDA device is visible and the problem is supported
function ok=matlab_gpu_supported(spin_system,n_outputs)
ok=false;
if ~rectangle_no_hooks(spin_system,n_outputs)
    return
end
try
    ok=(gpuDeviceCount>0);
catch
    ok=false;
end
end

% Reports whether the problem is a rectangle integration with no extra hooks
function ok=rectangle_no_hooks(spin_system,n_outputs)
ok=false;
if n_outputs>3
    return
end
if ~isfield(spin_system,'control')||...
   ~isfield(spin_system.control,'integrator')||...
   ~strcmp(spin_system.control.integrator,'rectangle')
    return
end
if isfield(spin_system.control,'prefix')&&~isempty(spin_system.control.prefix)
    return
end
if isfield(spin_system.control,'suffix')&&~isempty(spin_system.control.suffix)
    return
end
if isfield(spin_system.control,'dead_time')&&spin_system.control.dead_time~=0
    return
end
if isfield(spin_system.control,'keyholes')&&any(~cellfun(@isempty,spin_system.control.keyholes(:)))
    return
end
ok=true;
end

% Runs the CUDA backend against a cached persistent GPU operator pack
function varargout=cuda_backend(spin_system,drifts,controls,waveform,...
                                rho_init,rho_targ,fidelity_type,n_outputs)
persistent cache

% The cache starts empty and survives until the operators change
if isempty(cache)
    cache=struct('key',[],'handle',[]);
end

% Operators and the time grid together identify a pack
key={drifts,controls,spin_system.control.pulse_dt(:).'};
if isempty(cache.handle)||~isequal(cache.key,key)
    if ~isempty(cache.handle)
        try
            grape_liouv_gpu_mex('destroy',cache.handle);
        catch
        end
    end
    cache.handle=grape_liouv_gpu_mex('pack',drifts,controls,spin_system.control.pulse_dt);
    cache.key=key;
end

% Exact Frechet derivatives unless the caller asked for something cheaper
gradient_mode='exact';
if isfield(spin_system.control,'gpu_gradient')&&~isempty(spin_system.control.gpu_gradient)
    gradient_mode=spin_system.control.gpu_gradient;
end

% The forward trajectory is only returned when the plotting settings need it
return_forward=needs_forward_traj(spin_system);

% Populate requested outputs only
varargout=cell(1,n_outputs);
[varargout{1:n_outputs}]=grape_liouv_gpu_mex('eval',cache.handle,waveform,...
                                             rho_init,rho_targ,fidelity_type,...
                                             gradient_mode,return_forward);

end

% Rectangle integrator translation of grape_liouv.m with gpuArray operators
function varargout=gpuarray_backend(spin_system,drifts,controls,waveform,...
                                    rho_init,rho_targ,fidelity_type,n_outputs)

% Step.m must be told to use the GPU, and must not print anything
spin_system.sys.output='hush';
if ~isfield(spin_system.sys,'enable')||isempty(spin_system.sys.enable)
    spin_system.sys.enable={'gpu'};
elseif ~ismember('gpu',spin_system.sys.enable)
    spin_system.sys.enable{end+1}='gpu';
end

% Problem dimensions
nsteps=size(waveform,2); nctrls=size(waveform,1);
statedim=size(rho_init,1); ndrifts=numel(drifts);
dt=spin_system.control.pulse_dt;

% Static data goes up once per objective evaluation
drifts_g=cell(size(drifts));
for n=1:numel(drifts)
    drifts_g{n}=gpuArray(drifts{n});
end
controls_g=cell(size(controls));
for k=1:numel(controls)
    controls_g{k}=gpuArray(controls{k});
end
rho_init_g=gpuArray(full(rho_init));
rho_targ_g=gpuArray(full(rho_targ));
waveform_g=gpuArray(waveform);

% Trajectories are preallocated on the device
proto=gpuArray(complex(0));
fwd_traj=zeros(statedim,nsteps+1,'like',proto);
bwd_traj=zeros(statedim,nsteps+1,'like',proto);
fwd_traj(:,1)=rho_init_g; bwd_traj(:,1)=rho_targ_g;

% Forward and adjoint generators for every time slice
L_forw=cell(1,nsteps); L_back=cell(1,nsteps);
for n=1:nsteps
    L_forw{n}=drifts_g{mod(n-1,ndrifts)+1};
    L_back{n}=drifts_g{mod(nsteps-n,ndrifts)+1}';
    for k=1:nctrls
        L_forw{n}=L_forw{n}+waveform_g(k,n)*controls_g{k};
        L_back{n}=L_back{n}+waveform_g(k,nsteps+1-n)*controls_g{k}';
    end
end

% Forward trajectory from rho_init, backward trajectory from rho_targ
for n=1:nsteps
    fwd_traj(:,n+1)=step(spin_system,L_forw{n},fwd_traj(:,n),+dt(n));
    bwd_traj(:,n+1)=step(spin_system,L_back{n},bwd_traj(:,n),-dt(nsteps+1-n));
end

% Overlap of the propagated state with the target
overlap=rho_targ_g'*fwd_traj(:,end);

% Gradient by the auxiliary matrix formula used in grape_liouv.m
if n_outputs>2
    bwd_traj_flip=fliplr(bwd_traj);
    grad=zeros(size(waveform),'like',proto);
    zero_state=zeros(statedim,1,'like',proto);
    zero_drift=gpuArray(complex(spalloc(statedim,statedim,0)));
    for n=1:nsteps
        grad_col=zeros(nctrls,1,'like',proto);
        for k=1:nctrls
            aux_matrix=[L_forw{n}  controls_g{k}
                        zero_drift L_forw{n}    ];
            aux_vec=[zero_state; fwd_traj(:,n)];
            aux_vec=step(spin_system,aux_matrix,aux_vec,dt(n));
            grad_col(k)=bwd_traj_flip(:,n+1)'*aux_vec(1:(end/2));
        end
        grad(:,n)=grad_col;
    end
end

% Fidelity post-processing happens on the host, as in grape_liouv.m
overlap=gather(overlap);
if exist('grad','var')
    grad=gather(grad);
end

% Fidelity and gradient depend on which part of the overlap is wanted
switch fidelity_type
    case 'real'
        fidelity=real(overlap);
        if exist('grad','var'), grad=real(grad); end
    case 'imag'
        fidelity=imag(overlap);
        if exist('grad','var'), grad=imag(grad); end
    case 'square'
        fidelity=overlap*conj(overlap);
        if exist('grad','var')
            grad=grad*conj(overlap)+overlap*conj(grad);
            grad=real(grad);
        end
    otherwise
        error('unknown fidelity type');
end

% Trajectory return policy matches grape_liouv.m
traj_data.forward=[];
if needs_forward_traj(spin_system)
    traj_data.forward=gather(fwd_traj);
end

% Unreachable targets are a hard failure in the reference implementation
if abs(fidelity)==0
    spin_system.sys.output=1;
    report(spin_system,'exactly zero fidelity: either the target is unreachable');
    report(spin_system,'from the source, or the initial guess is very poor.');
    error('GRAPE cannot proceed.');
end
if exist('grad','var')&&(norm(grad,1)==0)
    spin_system.sys.output=1;
    report(spin_system,'exactly zero gradient: either the target is unreachable');
    report(spin_system,'from the source, or the initial guess is very poor.');
    error('GRAPE cannot proceed.');
end

% Populate requested outputs only
varargout=cell(1,n_outputs);
if n_outputs>=1, varargout{1}=traj_data; end
if n_outputs>=2, varargout{2}=fidelity; end
if n_outputs>=3, varargout{3}=grad; end

end

% Reports whether any diagnostic plot needs the forward trajectory
function tf=needs_forward_traj(spin_system)
tf=isfield(spin_system,'control')&&isfield(spin_system.control,'plotting')&&...
   any(ismember({'correlation_order','coherence_order',...
                 'local_each_spin','total_each_spin',...
                 'level_populations'},spin_system.control.plotting(:)));
end

% Consistency enforcement
function grumble(spin_system,drifts,controls,waveform,rho_init,rho_targ)
if ~isstruct(spin_system)
    error('spin_system must be a structure.');
end
if ~isfield(spin_system,'bas')||~isfield(spin_system.bas,'formalism')||...
   ~ismember(spin_system.bas.formalism,{'sphten-liouv','zeeman-liouv','zeeman-wavef'})
    error('this function requires a state vector based formalism.');
end
if (~isnumeric(rho_init))||(~iscolumn(rho_init))
    error('rho_init must be a column vector.');
end
if (~isnumeric(rho_targ))||(~iscolumn(rho_targ))
    error('rho_targ must be a column vector.');
end
if ~iscell(drifts)
    error('drifts must be a cell array of matrices.');
end
for n=1:numel(drifts)
    if (~isnumeric(drifts{n}))||(size(drifts{n},1)~=size(drifts{n},2))
        error('all elements of drifts cell array must be square matrices.');
    end
    if (size(drifts{n},1)~=numel(rho_init))||(size(drifts{n},1)~=numel(rho_targ))
        error('dimensions of drift, rho_init and rho_targ must be consistent.');
    end
end
if ~iscell(controls)
    error('controls must be a cell array of square matrices.');
end
for n=1:numel(controls)
    if (~isnumeric(controls{n}))||(size(controls{n},1)~=size(controls{n},2))||...
       (size(controls{n},1)~=size(drifts{1},1))
        error('control operators must have the same size as drift operators.');
    end
end
if (~isnumeric(waveform))||(~isreal(waveform))
    error('waveform must be a real numeric array.');
end
if size(waveform,1)~=numel(controls)
    error('number of waveform rows must be equal to the number of controls.');
end
if ~isfield(spin_system,'control')||~isfield(spin_system.control,'pulse_ntpts')||...
   size(waveform,2)~=spin_system.control.pulse_ntpts
    error('waveform length is inconsistent with spin_system.control.pulse_ntpts.');
end
if ~isfield(spin_system.control,'integrator')
    error('spin_system.control.integrator is missing.');
end
if strcmp(spin_system.control.integrator,'rectangle')&&...
   (size(spin_system.control.pulse_dt,2)~=size(waveform,2))
    error('pulse_dt must have the same length as waveform for rectangle integrator.');
end
if strcmp(spin_system.control.integrator,'trapezium')&&...
   (size(spin_system.control.pulse_dt,2)+1~=size(waveform,2))
    error('pulse_dt must be one element shorter than waveform for trapezium integrator.');
end
end

