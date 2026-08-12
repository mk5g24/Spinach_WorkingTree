function [r_avg, CSA_avg] = methyl_group_to_single_spin(positions, CSA_tensor, axis)

    % Calculate centre
    r_avg = 1/3*(positions{1} + positions{2} + positions{3});

    % Average CSA_tensors, norm the axis of rotation
    axis_n = axis/norm(axis);

    % Build the K matrix
    K = [0         -axis_n(3) axis_n(2);
        axis_n(3)  0          -axis_n(1);
        -axis_n(2) axis_n(1)  0;];
    K_squared = K*K;

    % Averaging number
    n_avg = 360;
    CSA_avg = cell(numel(CSA_tensor),1);
    for l = 1:numel(CSA_tensor)
        CSA_avg{l} = zeros(3,3);
    end

    for n=1:n_avg
        R = eye(3) + sin(n/n_avg*2*pi)*K + (1-cos(n/n_avg*2*pi))*K_squared;
        for l = 1:numel(CSA_tensor)
            CSA_avg{l} = CSA_avg{l} + R*CSA_tensor{l}*R';
        end
    end

     for l = 1:numel(CSA_tensor)
        CSA_avg{l} = CSA_avg{l}/n_avg;
    end   

end