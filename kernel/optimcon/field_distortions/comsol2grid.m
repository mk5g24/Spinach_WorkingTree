function B1f_obj = comsol2grid(filename,rotor,grid)
         
    rr = rotor.rotor_radius; % Rotor radius
    rl = rotor.rotor_length; % Rotor length
    cl = rotor.coil_length; % Coil length

    % Number of Extrapolation Points for angle
    Nphi = 128*4;

    % Number of desired points used for the calculations
    NWr = grid.nr;
    NWz = grid.nz;

    % Read the data
    data = readmatrix(filename);

    % Convert into lab frame
    dataLab=zeros(size(data(:,4:6)));
    dataLab(:,1) = data(:,4)*1/sqrt(3) + data(:,6)*sqrt(2/3);
    dataLab(:,2) = data(:,5);
    dataLab(:,3) = -data(:,4)*sqrt(2/3) + data(:,6)*sqrt(1/3);

    % Auxiliary Grid
    phi = linspace(0,2*pi,Nphi+1);
    phi = phi(1:(end-1));

    % Create the interpolant
    B1LabAmplX = scatteredInterpolant(data(:,1), data(:,2), data(:,3), real(dataLab(:,1)), 'natural');
    B1LabAmplY = scatteredInterpolant(data(:,1), data(:,2), data(:,3), real(dataLab(:,2)), 'natural');
    scale_factor = 1/sqrt(B1LabAmplX(rotor.coil_center(1),rotor.coil_center(2),rotor.coil_center(3))^2 + B1LabAmplY(rotor.coil_center(1),rotor.coil_center(2),rotor.coil_center(3))^2);

    % Calculate the relevant points and weights using Gaussian quadrature
    [xR,wR]=gaussleg(0*rr,rr,NWr-1);
    [xZ,wZ]=gaussleg(0.5*(cl-rl),0.5*(cl+rl),NWz-1);
    
    % Preallocate arrays for weights and points
    totalWeights = zeros(NWr,NWz);
    totalPoints = zeros(NWr,NWz,2);
    
    % Fill the weights
    for n=1:numel(xZ)
        for m=1:numel(xR)
            totalWeights(m,n) = xR(m)*wR(m)*wZ(n);
            totalPoints(m,n,1) = xR(m);
            totalPoints(m,n,2) = xZ(n);
        end
    end

    % Preallocate B1 field object
    % B1f_obj = cell(NWr*NWz,1);

    % Normalize the weights
    totalWeights = totalWeights*1/sum(totalWeights,'all');

    % Interpolate for a rotation around the z-Axis
    for n=1:NWz
        for m=1:NWr

            % Auxiliary Phi Array
            AuxPhi = zeros(Nphi);
            for k=1:Nphi; AuxPhi(k) = B1LabAmplX(totalPoints(m,n,1)*cos(phi(k)),totalPoints(m,n,1)*sin(phi(k)),totalPoints(m,n,2)); end

            % Extract significant values
            AuxPhiFFT = fft(AuxPhi);

            % Fill up the object
            B1f_obj.x(NWr*(n-1)+m) = scale_factor*AuxPhiFFT(1)*1/Nphi;
            B1f_obj.x_c1(NWr*(n-1)+m) = scale_factor*abs(AuxPhiFFT(2))*2/Nphi;
            B1f_obj.x_phi0(NWr*(n-1)+m) = atan2(-imag(AuxPhiFFT(2)),real(AuxPhiFFT(2)));

            % Auxiliary Phi Array
            AuxPhi = zeros(Nphi);
            for k=1:Nphi; AuxPhi(k) = B1LabAmplY(totalPoints(m,n,1)*cos(phi(k)),totalPoints(m,n,1)*sin(phi(k)),totalPoints(m,n,2)); end

            % Extract significant values
            AuxPhiFFT = fft(AuxPhi);
            B1f_obj.y(NWr*(n-1)+m) = scale_factor*AuxPhiFFT(1)*1/Nphi;
            B1f_obj.y_c1(NWr*(n-1)+m) = scale_factor*abs(AuxPhiFFT(2))*2/Nphi;
            B1f_obj.y_phi0(NWr*(n-1)+m) = atan2(-imag(AuxPhiFFT(2)),real(AuxPhiFFT(2)));

            % Set weights
            B1f_obj.weight(NWr*(n-1)+m) = totalWeights(m,n);

        end
    end

end