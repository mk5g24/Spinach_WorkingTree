% Converts a cell array of operators into sparse storage. The CUDA backend
% reads its operators through the MATLAB sparse interface, so anything that
% arrives dense has to be converted before it is packed onto the device.
% Syntax:
%
%     ops=sparsify_ops_gpu(ops)
%
% Parameters:
%
%   ops - cell array of matrices, in any MATLAB storage format
%
% Outputs:
%
%   ops - cell array of the same matrices in sparse storage
%
% Note: this is a conversion, not a truncation; no elements are dropped
%       and structural zeros of a dense input do not become stored zeros.
%
% m.keitel@soton.ac.uk

function ops=sparsify_ops_gpu(ops)

% Check consistency
grumble(ops);

% Sparse storage is what the device packer reads
for k=1:numel(ops)
    ops{k}=sparse(ops{k});
end

end

% Consistency enforcement
function grumble(ops)
if ~iscell(ops)
    error('ops must be a cell array of matrices.');
end
for k=1:numel(ops)
    if ~isnumeric(ops{k})
        error('all elements of the ops cell array must be numeric matrices.');
    end
end
end

