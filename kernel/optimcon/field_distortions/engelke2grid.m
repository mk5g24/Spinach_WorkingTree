% Transforms magnetic field values into proper frame

function B1f_obj = engelke2grid(filename)

    % Read the data
    output = load(filename);
    values = output.values;
    weights = output.weights;

    % Define phase and amplitude reference
    reference = output.reference;
    reference(:,1) = reference(:,1)*1/sqrt(3) + reference(:,3)*sqrt(2/3);

    % Get number of rotor rank and B1 field ensemble members
    N_B1 = numel(weights);

    % Convert into lab frame
    dataLab=zeros(size(values));
    dataLab(:,:,1) = values(:,:,1)*1/sqrt(3) + values(:,:,3)*sqrt(2/3);
    dataLab(:,:,2) = values(:,:,2);
    dataLab(:,:,3) = -values(:,:,1)*sqrt(2/3) + values(:,:,3)*sqrt(1/3);

    % Extract Amplitude and Phase
    B1_LA_X = abs(dataLab(:,:,1));
    B1_LA_Y = abs(dataLab(:,:,2));
    B1_LP_X = atan2(imag(dataLab(:,:,1)),real(dataLab(:,:,1)));
    B1_LP_Y = atan2(imag(dataLab(:,:,2)),real(dataLab(:,:,2)));

    ref_Ampl = sqrt(abs(reference(:,1))^2 + abs(reference(:,2))^2);
    ref_x0 = atan2(imag(reference(:,1)), real(reference(:,1)));
    
    % Interpolate for a rotation around the z-Axis
    wsum = 0;
    for n=1:N_B1

        B1f_obj(n).Cxx = cos(ref_x0 - B1_LP_X(n,:)).*B1_LA_X(n,:)/ref_Ampl - sin(ref_x0 - B1_LP_Y(n,:)).*B1_LA_Y(n,:)/ref_Ampl;
        B1f_obj(n).Cxy = -sin(ref_x0 - B1_LP_X(n,:)).*B1_LA_X(n,:)/ref_Ampl - cos(ref_x0 - B1_LP_Y(n,:)).*B1_LA_Y(n,:)/ref_Ampl;
        B1f_obj(n).Cyx = sin(ref_x0 - B1_LP_X(n,:)).*B1_LA_X(n,:)/ref_Ampl + cos(ref_x0 - B1_LP_Y(n,:)).*B1_LA_Y(n,:)/ref_Ampl;
        B1f_obj(n).Cyy = cos(ref_x0 - B1_LP_X(n,:)).*B1_LA_X(n,:)/ref_Ampl - sin(ref_x0 - B1_LP_Y(n,:)).*B1_LA_Y(n,:)/ref_Ampl;

        % Detection Sensitivity of the Different Coil Regions
        detection_weight = real(sum(B1_LA_X(n,:).*cos(ref_x0 - B1_LP_X(n,:))) - 1j*sum(B1_LA_Y(n,:).*sin(ref_x0 - B1_LP_X(n,:))))/ref_Ampl;

        B1f_obj(n).weight = detection_weight*weights(n);
        wsum = wsum + B1f_obj(n).weight;

    end

    % Normalise the weights
    for n=1:N_B1
        B1f_obj(n).weight = B1f_obj(n).weight/wsum; 
    end

end