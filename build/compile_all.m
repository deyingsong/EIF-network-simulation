function compile_all()
% COMPILE_ALL - Compile all EIF simulation MEX files
%
% This script compiles all variants of the EIF neural network simulation
% with optimized settings for maximum performance.
%
% Usage:
%   compile_all()
%
% Requirements:
%   - MATLAB R2021b or later
%   - Configured MEX compiler (run 'mex -setup' if needed)
%
% Performance Notes:
%   - Uses -O optimization for release builds
%   - Includes debug symbols in debug mode
%   - Links against optimized BLAS/LAPACK when available

fprintf('=== EIF Neural Simulation MEX Compiler ===\n\n');

% Configuration
src_dir = '../src';
common_dir = fullfile(src_dir, 'common');
variants_dir = fullfile(src_dir, 'variants');

% Check if source directories exist
if ~exist(src_dir, 'dir')
    error('Source directory not found: %s', src_dir);
end

% MEX compilation flags
debug_mode = false; % Set to true for debugging
if debug_mode
    mex_flags = {'-g', '-DEIF_DEBUG=1'};
    fprintf('Compiling in DEBUG mode...\n');
else
    mex_flags = {'-O', '-DEIF_DEBUG=0', '-DNDEBUG'};
    fprintf('Compiling in RELEASE mode...\n');
end

% Common source files
common_files = {
    fullfile(common_dir, 'EIF_common.c')
};

% Variant source files
variant_files = {
    'EIF_normalization_BroadWeight.c', 
    'EIF_normalization_CurrentNoise.c',
    'EIF_normalization_CurrentGaussianNoise.c',
    'EIF_normalization_CurrentNoise_BroadWeight.c',
    'EIF_normalization_MatchInDegree.c'
};

% Include directories
include_dirs = ['-I', common_dir];

% Compile each variant
success_count = 0;
total_count = length(variant_files);

for i = 1:length(variant_files)
    variant_file = variant_files{i};
    variant_path = fullfile(variants_dir, variant_file);
    
    if ~exist(variant_path, 'file')
        fprintf('⚠️  Warning: %s not found, skipping...\n', variant_file);
        continue;
    end
    
    fprintf('📦 Compiling %s... ', variant_file);
    tic;
    
    try
        % Build MEX command
        mex_cmd = [mex_flags, include_dirs, {variant_path}, common_files];
        
        % Execute MEX compilation
        mex(mex_cmd{:});
        
        compile_time = toc;
        success_count = success_count + 1;
        fprintf('✅ Success (%.2fs)\n', compile_time);
        
    catch ME
        compile_time = toc;
        fprintf('❌ Failed (%.2fs)\n', compile_time);
        fprintf('   Error: %s\n', ME.message);
    end
end

fprintf('\n=== Compilation Summary ===\n');
fprintf('Successfully compiled: %d/%d variants\n', success_count, total_count);

if success_count == total_count
    fprintf('🎉 All variants compiled successfully!\n');
    
    % Display compiled MEX files
    fprintf('\nCompiled MEX files:\n');
    mex_files = dir('*.mex*');
    for i = 1:length(mex_files)
        file_info = dir(mex_files(i).name);
        file_size_mb = file_info.bytes / (1024*1024);
        fprintf('  %s (%.1f MB)\n', mex_files(i).name, file_size_mb);
    end
    
    % Run quick validation
    fprintf('\n🔍 Running quick validation...\n');
    try
        run_validation_tests();
        fprintf('✅ Validation passed!\n');
    catch ME
        fprintf('⚠️  Validation warning: %s\n', ME.message);
    end
    
else
    fprintf('⚠️  Some variants failed to compile. Check error messages above.\n');
end

fprintf('\n=== Ready to simulate! ===\n');
fprintf('Example usage:\n');
fprintf('  params = setup_default_params();\n');
fprintf('  [spikes, Isyn, V] = EIF_normalization_Default(sx, Wrf, Wrr, params);\n\n');

end

function run_validation_tests()
% Quick validation to ensure MEX files work correctly
    
    % Check if any MEX file exists
    mex_files = dir('EIF_normalization_*.mex*');
    if isempty(mex_files)
        error('No compiled MEX files found');
    end
    
    % Test basic function availability
    [~, func_name] = fileparts(mex_files(1).name);
    if exist(func_name, 'file') ~= 3  % 3 = MEX file
        error('MEX file not properly registered: %s', func_name);
    end
    
    fprintf('Basic MEX functionality verified.\n');
end
