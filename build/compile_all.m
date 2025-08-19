function compile_all(varargin)
% compile_all  Build all normalization MEX files for the current platform.
%
% Folder layout assumed:
%   build/compile_all.m
%   src/common/EIF_common.c, EIF_common.h
%   src/variants/EIF_normalization_*.c
%
% Behavior:
%   - Automatically detects whether a variant includes "EIF_common.h".
%     If yes, it compiles & links with src/common/EIF_common.c and adds -Isrc/common.
%   - Treats EIF_normalization_Default.c as self‑contained (no common files).
%   - Produces one MEX per variant with the same base name as the .c file.

clc;

thisDir     = fileparts(mfilename('fullpath'));
root        = fileparts(thisDir);
commonDir   = fullfile(root,'src','common');
variantsDir = fullfile(root,'src','variants');

% ---- variants (update here if you add more files) -----------------------
variants = { ...
    'EIF_normalization_BroadWeight' ...
    'EIF_normalization_CurrentGaussianNoise' ...
    'EIF_normalization_CurrentNoise' ...
    'EIF_normalization_CurrentNoise_BroadWeight' ...
    'EIF_normalization_MatchInDegree' ...
    'EIF_normalization_Default' ...
};

% ---- platform/compiler flags -------------------------------------------
mexArgsCommon = {};
if ispc
    % MSVC
    mexArgsCommon = {'-largeArrayDims', 'COMPFLAGS=$COMPFLAGS /O2'};
elseif ismac
    % Clang
    mexArgsCommon = {'-largeArrayDims', 'CFLAGS=$CFLAGS -O3 -std=c11'};
else
    % GCC/Clang on Linux
    mexArgsCommon = {'-largeArrayDims', 'CFLAGS=$CFLAGS -O3 -std=c11'};
end

% Optional: allow passing extra flags from command line, e.g. compile_all('-v')
mexArgsCommon = [mexArgsCommon, varargin];

fprintf('Building MEX files...\n');
fprintf('Root:      %s\n', root);
fprintf('Common:    %s\n', commonDir);
fprintf('Variants:  %s\n\n', variantsDir);

built = {};
failed = {};

for k = 1:numel(variants)
    name   = variants{k};
    srcC   = fullfile(variantsDir, [name '.c']);
    output = name;

    if ~exist(srcC, 'file')
        warning('Skipping %s (source not found: %s)', name, srcC);
        failed{end+1} = name; %#ok<AGROW>
        continue;
    end

    % Heuristic: detect whether this variant uses the shared common code.
    % We *always* treat Default as standalone per your note.
    usesCommon = ~strcmpi(name, 'EIF_normalization_Default');
    try
        txt = fileread(srcC);
        if ~contains(txt, 'EIF_common.h')
            usesCommon = false;
        end
    catch
        % If fileread fails, fall back to the filename rule above
    end

    % Build the mex command
    try
        if usesCommon
            fprintf('> %s (with common)\n', name);
            mex(mexArgsCommon{:}, ...
                ['-I' commonDir], ...
                fullfile(commonDir, 'EIF_common.c'), ...
                srcC, ...
                '-output', output);
        else
            fprintf('> %s (standalone)\n', name);
            mex(mexArgsCommon{:}, ...
                srcC, ...
                '-output', output);
        end
        built{end+1} = name; %#ok<AGROW>
    catch ME
        failed{end+1} = name; %#ok<AGROW>
        fprintf(2, '  FAILED: %s\n  %s\n\n', name, ME.message);
    end
end

fprintf('\n=== Summary ===\n');
fprintf('Built:  %s\n', strjoin(built, ', '));
if ~isempty(failed)
    fprintf(2, 'Failed: %s\n', strjoin(failed, ', '));
end
fprintf('Done.\n');

end
