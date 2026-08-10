% t - Time points
% B1 - cell array of B1 field strengths
% c1 - cell array of first term coefficients
% phi0 - cell array of phase offset

function dist_funcs = MASmodulation(varargin)

% Always use these
t =  varargin{1};
wMAS =  varargin{2};
B1_obj = varargin{3};

if numel(varargin)>3
    phi = varargin{4};
else
    phi = 0;
end

% Get number of control channels
nctrlchannels = numel(B1_obj);

% Initialize Matrix
transormation_matrix = zeros(2*nctrlchannels,2*nctrlchannels,numel(t));

% Preallocate output
n_weights = numel(B1_obj{1}.weight);
dist_funcs = cell(n_weights,1);

for k = 1:n_weights
    for n = 1:nctrlchannels
        
        % Diagonal terms
        transormation_matrix(2*n-1,2*n-1,:) = B1_obj{n}.x(k)*(1+B1_obj{n}.x_c1(k)*cos(2*pi*wMAS*t-B1_obj{n}.x_phi0(k)-phi));
        transormation_matrix(2*n,2*n,:) = B1_obj{n}.x(k)*(1+B1_obj{n}.x_c1(k)*cos(2*pi*wMAS*t-B1_obj{n}.x_phi0(k)-phi));
    
        % Offdiagonal terms
        transormation_matrix(2*n-1,2*n,:) = -B1_obj{n}.y(k)*(1+B1_obj{n}.y_c1(k)*cos(2*pi*wMAS*t-B1_obj{n}.y_phi0(k)-phi));
        transormation_matrix(2*n,2*n-1,:) = B1_obj{n}.y(k)*(1+B1_obj{n}.y_c1(k)*cos(2*pi*wMAS*t-B1_obj{n}.y_phi0(k)-phi));
    end

    % Build up the Jacobian from the transformation matrix
    row = zeros(2*nctrlchannels*numel(t),1);
    col = zeros(2*nctrlchannels*numel(t),1);
    val = zeros(2*nctrlchannels*numel(t),1);
    index = 1;

    for n=1:2*numel(t)*nctrlchannels
        for m=1:2*numel(t)*nctrlchannels
            a = floor((n-1)/(2*nctrlchannels))+1;
            b = mod(n-1,2*nctrlchannels)+1;
            c = floor((m-1)/(2*nctrlchannels))+1;
            d = mod(m-1,2*nctrlchannels)+1;
            if a==c
                row(index) = n;
                col(index) = m;
                val(index) = transormation_matrix(b,d,a);
                index = index + 1;
            end
        end
    end
    
    jacobian = sparse(row, col, val);

    % Build the function handle
    dist_funcs{k} = @(wf) (dst_fc(transormation_matrix,wf,jacobian));
end

end

function [dstrd_wfm,jacobian] = dst_fc(mat,wf,jacobian)

    dstrd_wfm = zeros(size(wf));

    for n=1:size(mat,3)
        dstrd_wfm(:,n) = mat(:,:,n)*wf(:,n);
    end

end