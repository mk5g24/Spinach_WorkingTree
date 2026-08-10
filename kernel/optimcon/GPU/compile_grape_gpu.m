% Compiles the CUDA/MEX backend of the GPU Liouville-space GRAPE engine
% into grape_liouv_gpu_mex.<mexext> in the directory that holds this file.
% The compute capability of every visible GPU is queried, and device code
% is emitted for each of them, followed by forward-compatible PTX for the
% newest one. This replaces the GNU makefile and works on any platform on
% which mexcuda is functional. The result is loaded and its capability
% query is inspected before this function returns. Syntax:
%
%     compile_grape_gpu(host_compiler)
%
% Parameters:
%
%    host_compiler - 'default' to let the CUDA toolkit pick its
%                    own host compiler, or the full path to a
%                    C++ compiler executable that the toolkit
%                    accepts; CUDA 12.x front ends cannot parse
%                    GCC 14 and newer standard library headers,
%                    and need to be pointed at GCC 13 or older
%
% Outputs:
%
%    this function returns nothing; it writes the compiled MEX
%    file next to itself and prints a build report
%
% Note: grape_liouv_gpu.m only trusts a binary whose capability query
%       reports cudaBackendReady=true, and silently falls back to the
%       CPU otherwise, so that flag is checked here and a build that
%       fails to set it is reported as an error rather than a success.
%
% m.keitel@soton.ac.uk

function compile_grape_gpu(host_compiler)

% Check consistency
grumble(host_compiler);

% Sources and output live next to this file
code_dir=fileparts(mfilename('fullpath'));
source_file=fullfile(code_dir,'grape_liouv_gpu_cuda_mex.cu');
mex_name='grape_liouv_gpu_mex';
mex_file=fullfile(code_dir,[mex_name '.' mexext]);

% The CUDA source is the only build input
if ~isfile(source_file)
    error(['CUDA source not found: ' source_file]);
end

% Device code is emitted for the hardware that is actually present
if gpuDeviceCount==0
    error('no CUDA device is visible, cannot determine the target architecture.');
end
device_table=gpuDeviceTable;
arch_codes=unique(erase(string(device_table.ComputeCapability),'.'));
arch_codes=sort(arch_codes);

% Emit a cubin per architecture, plus PTX for future hardware
nvcc_flags="NVCCFLAGS=$NVCCFLAGS -Wno-deprecated-gpu-targets";
for n=1:numel(arch_codes)
    nvcc_flags=nvcc_flags+" -gencode=arch=compute_"+arch_codes(n)+",code=sm_"+arch_codes(n);
end
nvcc_flags=nvcc_flags+" -gencode=arch=compute_"+arch_codes(end)+",code=compute_"+arch_codes(end);

% An explicit host compiler is passed through to nvcc
if ~strcmp(host_compiler,'default')
    if ~isfile(host_compiler)
        error(['host compiler not found: ' host_compiler]);
    end
    nvcc_flags=nvcc_flags+" -ccbin "+host_compiler;
end

% Glibc 2.43 exposes C23 rsqrt under _GNU_SOURCE and clashes with CUDA
if isunix
    nvcc_flags=nvcc_flags+" --compiler-options=-U_GNU_SOURCE,-D_DEFAULT_SOURCE,-D_XOPEN_SOURCE=700";
end

% A loaded MEX file cannot be overwritten on Windows
clear(mex_name);

% Report the configuration before the compiler runs
fprintf('Compiling %s\n',source_file);
fprintf('  target architectures: %s\n',strjoin(cellstr(arch_codes),', '));
fprintf('  host compiler:        %s\n',host_compiler);

% Build the backend
mexcuda('-R2018a','-O',char(nvcc_flags),source_file,'-output',mex_name,'-outdir',code_dir);

% A silent mexcuda failure would leave no binary behind
if ~isfile(mex_file)
    error(['mexcuda reported no error but produced no ' mex_file]);
end

% An incomplete backend must not be reported as a working one
capabilities=grape_liouv_gpu_mex('capabilities');
if ~capabilities.cudaBackendReady
    error('the compiled backend does not report cudaBackendReady, it will not be used.');
end

% Report what the binary says about itself
fprintf('Built %s\n',mex_file);
fprintf('  backend: %s\n',capabilities.name);
fprintf('  version: %d\n',capabilities.version);
fprintf('  devices: %d\n',capabilities.deviceCount);

end

% Consistency enforcement
function grumble(host_compiler)
if ~ischar(host_compiler)
    error('host_compiler must be a character string.');
end
if isempty(host_compiler)
    error('host_compiler must not be empty, use ''default'' to let the toolkit choose.');
end
end

% The first principle is that you must not fool yourself,
% and you are the easiest person to fool.
%
% Richard Feynman

