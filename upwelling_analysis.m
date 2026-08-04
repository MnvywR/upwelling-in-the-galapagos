%% upwelling_analysis.m
%
% Positive-only (upwelling) diagnostics from Oceananigans xy-slice NetCDF
% output, at three fixed depths (75 m, 150 m, 225 m).
%
% For each depth level this produces, side by side:
%   LEFT  : time-averaged spatial map of w+ = max(w,0)          [m/s]
%   RIGHT : cumulative area-integrated upward volume flux        [m^3]
%           V(t) = integral_0^t ( integral_A max(w,0) dA ) dt'
%           which is monotonically non-decreasing by construction,
%           since the integrand is always >= 0.
%
% Requires: NetCDF files produced by your xy_75_depth_writer,
%           xy_150_depth_writer, xy_225_depth_writer.
%
% NOTE ON VARIABLE NAMES:
%   Oceananigans' NetCDFWriter names coordinate dimensions after the
%   field's grid location: cell-centered -> xC/yC/zC, face-centered ->
%   xF/yF/zF. The variable "w" lives on (Center, Center, Face), so its
%   horizontal coords are typically xC/yC. If any of the ncread calls
%   below error with "variable not found", run:
%       ncdisp('upwelling_75m.nc')
%   once to see the exact names in your file and edit the strings below.

clear; clc; close all;

%% ---- USER SETTINGS ----
depths       = [75, 150, 225];


data_dir = 'C:\External Programs and Downloads\Galapagos Project\runs\real_bathymetry_beta_no_wind_20260727_112143_997';

files = { fullfile(data_dir, 'upwelling_75m.nc'), ...
          fullfile(data_dir, 'upwelling_150m.nc'), ...
          fullfile(data_dir, 'upwelling_225m.nc') };


%% ---- USER SETTINGS ----
% Point this at the specific run folder Julia wrote to, e.g.
% ".../runs/real_bathymetry_beta_wind_20260731_143022_501"
% (leave as '.' if the .nc files are in the same folder as this script)
data_dir     = '.';

month_len    = 30.44;      % days per "month" tick, adjust if you prefer 30
use_common_clim = true;    % force identical color scale across depths for comparison
outfile      = 'upwelling_summary.png';

% Optional: apply a wet/dry mask if you exported bathymetry separately.
% Set this to a function handle that returns a logical mask the same
% size as w(:,:,1), or leave empty to skip masking.
wet_mask_fun = [];   % e.g. @(d) load_wet_mask_for_depth(d);

nDepths = numel(depths);

%% ---- PROCESS EACH DEPTH ----
results = struct('depth', {}, 'xC', {}, 'yC', {}, 'time_days', {}, ...
                  'w_pos_tmean', {}, 'Q', {}, 'V_cum', {});

for d = 1:nDepths
    fname = files{d};
    r = process_depth(fname, wet_mask_fun, depths(d));
    results(d) = r; %#ok<SAGROW>
end

%% ---- COMMON COLOR LIMITS (optional) ----
if use_common_clim
    all_vals = [];
    for d = 1:nDepths
        all_vals = [all_vals; results(d).w_pos_tmean(:)]; %#ok<AGROW>
    end
    clim_max = max(all_vals, [], 'omitnan');
    if clim_max <= 0, clim_max = eps; end
    common_clim = [0, clim_max];
else
    common_clim = [];
end

%% ---- PLOT: rows = depths, cols = [contour, cumulative flux] ----
fig = figure('Position', [100, 100, 1000, 300*nDepths], 'Color', 'w');
tl = tiledlayout(fig, nDepths, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for d = 1:nDepths
    r = results(d);

    % ---- LEFT: time-averaged spatial map of w+ ----
    ax1 = nexttile(tl, (d-1)*2 + 1);
    contourf(ax1, r.xC/1000, r.yC/1000, r.w_pos_tmean', 20, 'LineColor', 'none');
    colormap(ax1, parula);
    cb = colorbar(ax1);
    cb.Label.String = 'Time-mean w^+  [m/s]';
    if ~isempty(common_clim)
        clim(ax1, common_clim);
    end
    xlabel(ax1, 'x  [km]');
    ylabel(ax1, 'y  [km]');
    title(ax1, sprintf('%d m depth — mean positive-only w', r.depth));
    axis(ax1, 'equal', 'tight');

    % ---- RIGHT: cumulative upward volume flux over time ----
    ax2 = nexttile(tl, (d-1)*2 + 2);
    plot(ax2, r.time_days, r.V_cum, 'LineWidth', 1.6, 'Color', [0.10 0.40 0.75]);
    hold(ax2, 'on');
    month_ticks = 0:month_len:max(r.time_days);
    xticks(ax2, month_ticks);
    xticklabels(ax2, compose('%d mo', round(month_ticks/month_len)));
    grid(ax2, 'on');
    xlabel(ax2, 'Time');
    ylabel(ax2, 'Cumulative upward volume  [m^3]');
    title(ax2, sprintf('%d m depth — cumulative upwelling flux', r.depth));
end

sgtitle(tl, 'Positive-only (upwelling) diagnostics by depth', ...
    'FontWeight', 'bold');

exportgraphics(fig, outfile, 'Resolution', 300);
fprintf('Saved figure to %s\n', outfile);

%% =====================================================================
function r = process_depth(fname, wet_mask_fun, depth_label)
% Read one xy-slice NetCDF file and compute the two diagnostics.
% Coordinate/time dimension names are auto-detected from the "w"
% variable's actual dimensions, since Oceananigans versions differ in
% whether they write "xC"/"yC" or the location-suffixed form
% (e.g. "xᶜᵃᵃ", "yᵃᶜᵃ"). This avoids hardcoding a name that may not
% match your file.

    info = ncinfo(fname);
    allVarNames = {info.Variables.Name};
    wIdx = find(strcmp(allVarNames, 'w'), 1);
    if isempty(wIdx)
        error('upwelling_analysis:noW', ...
            'Variable "w" not found in %s. Variables present: %s', ...
            fname, strjoin(allVarNames, ', '));
    end
    dimNames = {info.Variables(wIdx).Dimensions.Name};

    isX = cellfun(@(s) startsWith(s, 'x'), dimNames);
    isY = cellfun(@(s) startsWith(s, 'y'), dimNames);
    isT = cellfun(@(s) strcmpi(s, 'time') || contains(lower(s), 'time'), dimNames);

    if ~any(isX) || ~any(isY) || ~any(isT)
        error('upwelling_analysis:dimsNotFound', ...
            ['Could not identify x/y/time dimensions for "w" in %s.\n' ...
             'Dimensions found: %s\nRun ncdisp(''%s'') and check manually.'], ...
            fname, strjoin(dimNames, ', '), fname);
    end

    xDimName = dimNames{isX};
    yDimName = dimNames{isY};
    timeDimName = dimNames{isT};

    fprintf('%s -> using x="%s", y="%s", time="%s"\n', ...
        fname, xDimName, yDimName, timeDimName);

    % ---- Coordinates & time ----
    xC   = ncread(fname, xDimName);    % [m]
    yC   = ncread(fname, yDimName);    % [m]
    time = double(ncread(fname, timeDimName));  % [s]

    % ---- Vertical velocity ----
    w = ncread(fname, 'w');        % (Nx, Ny, 1, Nt) or similar
    w = squeeze(w);                % drop singleton z -> (Nx, Ny, Nt)

    Nt = numel(time);
    if ndims(w) == 2   % single time step edge case
        w = reshape(w, size(w,1), size(w,2), 1);
    end

    % Guard against dimension order surprises (e.g. y before x)
    Nx = numel(xC); Ny = numel(yC);
    if size(w,1) ~= Nx || size(w,2) ~= Ny
        if size(w,1) == Ny && size(w,2) == Nx
            w = permute(w, [2 1 3]);
        else
            error('upwelling_analysis:sizeMismatch', ...
                ['w has size [%d %d %d] but xC/yC have lengths %d/%d.\n' ...
                 'Inspect ncinfo(''%s'') to confirm dimension order.'], ...
                size(w,1), size(w,2), size(w,3), Nx, Ny, fname);
        end
    end

    % ---- Optional wet/dry mask ----
    if ~isempty(wet_mask_fun)
        mask = wet_mask_fun(depth_label);   % logical (Nx,Ny)
        w(~repmat(mask, 1, 1, Nt)) = NaN;
    end

    % ---- Grid metrics ----
    dx = mean(diff(xC));
    dy = mean(diff(yC));
    dA = dx * dy;   % m^2, uniform Cartesian grid

    % ---- Positive-only vertical velocity ----
    w_pos = max(w, 0);   % NaNs (masked land) stay NaN

    % ---- (1) Time-averaged spatial field ----
    w_pos_tmean = mean(w_pos, 3, 'omitnan');   % (Nx, Ny)

    % ---- (2) Instantaneous area-integrated flux, then cumulative in time ----
    Q = squeeze(sum(sum(w_pos, 1, 'omitnan'), 2, 'omitnan')) * dA;  % (Nt,1) m^3/s
    Q = Q(:);
    V_cum = cumtrapz(time, Q);   % m^3, monotonically non-decreasing since Q >= 0

    % ---- Package ----
    r.depth       = depth_label;
    r.xC          = xC;
    r.yC          = yC;
    r.time_days   = time / 86400;
    r.w_pos_tmean = w_pos_tmean;
    r.Q           = Q;
    r.V_cum       = V_cum;
end