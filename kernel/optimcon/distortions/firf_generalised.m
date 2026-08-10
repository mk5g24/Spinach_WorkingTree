% Applies channel-specific FIR convolution filters to a Spinach
% optimal control module waveform. Treats odd rows of the wave-
% form as real, and even rows as imaginary, components of a
% complex signal. Each physical control channel receives its
% own FIR kernel. The distal end of each convolution is truncated
% to the original waveform length. Syntax:
%
%             [w,J]=firf_generalised(w,kernels)
%
% Parameters:
%
%    w        - waveform, one time slice per column, with rows
%               arranged as XYXY... with respect to in-phase
%               and quadrature components on each physical
%               control channel
%
%    kernels  - cell array of FIR coefficient vectors, one
%               vector per physical control channel
%
% Outputs:
%
%    w        - distorted waveform, same dimension as the input
%               waveform; leaving sufficient ring-down margin is
%               the user's responsibility
%
%    J        - Jacobian matrix with respect to vectorisations of
%               the output and the input arrays
%
% This function calls Spinach firf.m for each physical channel.
%

function [w,J]=firf_generalised(w,kernels)

% Check consistency
grumble(w,kernels);

% Count physical channels
nchans=size(w,1)/2;

% Preallocate the output waveform
w_dist=zeros(size(w),'like',w);

% Prepare the global index map
idx=reshape(1:numel(w),size(w));

% Preallocate the Jacobian
if nargout>1
    jac_rows=cell(nchans,1);
    jac_cols=cell(nchans,1);
    jac_vals=cell(nchans,1);
end

% Loop over physical channels
for n=1:nchans

    % Select the current XY row pair
    rows=2*n-1:2*n;

    % Apply the channel-specific FIR filter
    if nargout>1
        [w_chan,J_chan]=firf(w(rows,:),kernels{n});
    else
        w_chan=firf(w(rows,:),kernels{n});
    end

    % Store the distorted XY row pair
    w_dist(rows,:)=w_chan;

    % Insert the channel Jacobian into the global layout
    if nargout>1
        glob_idx=idx(rows,:);
        glob_idx=glob_idx(:);
        [loc_rows,loc_cols,loc_vals]=find(J_chan);
        jac_rows{n}=glob_idx(loc_rows);
        jac_cols{n}=glob_idx(loc_cols);
        jac_vals{n}=loc_vals;
    end

end

% Assemble the global Jacobian
if nargout>1
    J=sparse(vertcat(jac_rows{:}),vertcat(jac_cols{:}),...
             vertcat(jac_vals{:}),numel(w),numel(w));
end

% Return the distorted waveform
w=w_dist;

end

% Consistency enforcement
function grumble(w,kernels)
if (~isnumeric(w))||(~isreal(w))
    error('w must be an array of real numbers.');
end
if mod(size(w,1),2)~=0
    error('the number of rows in w must be even.');
end
if ~iscell(kernels)
    error('kernels must be a cell array.');
end
if numel(kernels)~=size(w,1)/2
    error('kernels must contain one vector per physical channel.');
end
for n=1:numel(kernels)
    if (~isnumeric(kernels{n}))||(~isvector(kernels{n}))
        error('each element of kernels must be a vector.');
    end
    if isempty(kernels{n})
        error('each element of kernels must have at least one element.');
    end
end
end
