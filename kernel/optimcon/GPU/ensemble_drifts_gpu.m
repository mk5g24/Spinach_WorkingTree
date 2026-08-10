% Assembles the ensemble of drift Liouvillians that the GPU GRAPE backend
% must average over. The drift generators themselves are collected from the
% parallel pool ValueStore where optimcon.m left them, and every requested
% resonance offset combination is then added to each of them. The result is
% a flat cell array with one entry per ensemble member, which is what the
% CUDA ensemble pack expects. Syntax:
%
%     ensemble_drifts=ensemble_drifts_gpu(spin_system)
%
% Parameters:
%
%   spin_system     - Spinach data object that has been through
%                     the optimcon.m problem setup function; its
%                     control.ndrifts, control.offsets, and cont-
%                     rol.off_ops fields are used here
%
% Outputs:
%
%   ensemble_drifts - cell array with one member per combination
%                     of drift generator and offset; each member
%                     is either a matrix or, for time-dependent
%                     drift, a cell array of matrices
%
% Note: offsets follow the convention of Spinach ensemble.m, where they
%       are specified in Hz for each control channel and enter the drift
%       as 2*pi*offset*off_op. Offsets across channels are combined in
%       all possible ways, so the ensemble grows as the product of the
%       number of drifts and the offset counts of every channel.
%
% Note: this function requires an active parallel pool, because that is
%       where optimcon.m stores the drift generators. Run optimcon.m
%       before calling it.
%
% m.keitel@soton.ac.uk

function ensemble_drifts=ensemble_drifts_gpu(spin_system)

% Check consistency
grumble(spin_system);

% Drift generators are left in the pool ValueStore by optimcon.m
pool=gcp('nocreate');
if isempty(pool)
    error('no parallel pool is active; run optimcon() first so the drift ValueStore exists.');
end
store=pool.ValueStore;
base_drifts=cell(1,spin_system.control.ndrifts);
for n=1:spin_system.control.ndrifts
    key=['oc_drift_' num2str(n)];
    try
        base_drifts{n}=store(key);
    catch ME
        error('could not retrieve %s from the parallel-pool ValueStore: %s',key,ME.message);
    end
end

% Offset operators are the outer product of the per-channel offset lists
if isempty(spin_system.control.offsets)
    offset_ops={[]};
else
    n=drift_dimension(base_drifts{1});
    offset_ops={sparse(n,n)};
    for ch=1:numel(spin_system.control.offsets)
        next_ops=cell(1,numel(offset_ops)*numel(spin_system.control.offsets{ch}));
        pos=0;
        for a=1:numel(offset_ops)
            for b=1:numel(spin_system.control.offsets{ch})
                pos=pos+1;
                next_ops{pos}=offset_ops{a}+sparse(2*pi*spin_system.control.offsets{ch}(b)*spin_system.control.off_ops{ch});
            end
        end
        offset_ops=next_ops;
    end
end

% Every drift generator is paired with every offset combination
ensemble_drifts=cell(1,numel(base_drifts)*numel(offset_ops));
pos=0;
for d=1:numel(base_drifts)
    for o=1:numel(offset_ops)
        pos=pos+1;
        ensemble_drifts{pos}=add_offset_to_drift(base_drifts{d},offset_ops{o});
    end
end

end

% Returns the state space dimension of a drift that may be time-dependent
function n=drift_dimension(drift)
if iscell(drift)
    n=size(drift{1},1);
else
    n=size(drift,1);
end
end

% Adds an offset operator to a drift that may be time-dependent
function out=add_offset_to_drift(drift,offset_op)
if isempty(offset_op)
    out=drift;
elseif iscell(drift)
    out=drift;
    for k=1:numel(drift)
        out{k}=sparse(drift{k})+offset_op;
    end
else
    out=sparse(drift)+offset_op;
end
end

% Consistency enforcement
function grumble(spin_system)
if (~isstruct(spin_system))||(~isfield(spin_system,'control'))
    error('spin_system must be a Spinach object processed by optimcon().');
end
if ~isfield(spin_system.control,'ndrifts')
    error('spin_system.control.ndrifts is missing, run optimcon() first.');
end
if ~isfield(spin_system.control,'offsets')
    error('spin_system.control.offsets is missing, run optimcon() first.');
end
if (~isempty(spin_system.control.offsets))&&(~isfield(spin_system.control,'off_ops'))
    error('spin_system.control.off_ops is missing, but offsets are specified.');
end
end

