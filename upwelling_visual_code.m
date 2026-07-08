%% ==========================================
% Upwelling Velocity Visualization
% xy slices at 75 m, 150 m, 225 m
%% ==========================================

%% ---- File locations ----
data_folder = 'C:/External Programs and Downloads/Galapagos Project/runs/real_bathymetry_beta_no_wind_20260707_175743_004';

file_75  = fullfile(data_folder, 'upwelling_75m.nc');
file_150 = fullfile(data_folder, 'upwelling_150m.nc');
file_225 = fullfile(data_folder, 'upwelling_225m.nc');

files  = {file_75, file_150, file_225};
depths = [75, 150, 225];
ndep   = numel(files);

%% ---- Variable name for vertical (upwelling) velocity ----
% Confirmed via ncdisp: 'w', dims (x_caa, y_aca, z_aaf, time)
var_name = 'w';

%% ---- Load coordinates (assumed same grid/time across all three files) ----
x    = ncread(file_75, 'x_caa')/1000;   % km
y    = ncread(file_75, 'y_aca')/1000;   % km
time = ncread(file_75, 'time');
nt   = length(time);

%% ---- Land masks (static, one per depth file) ----
% peripheral_nodes_ccc flags immersed/land cells: nonzero = land, 0 = fluid
land_masks = cell(ndep, 1);
for id = 1:ndep
    pn = squeeze(ncread(files{id}, 'peripheral_nodes_ccc'));  % 30x30
    land_masks{id} = (pn ~= 0)';   % transpose to match pcolor(x,y,W') orientation
end

%% ==========================================
% AUTO-SCAN: robust, depth-specific color limits
% (symmetric about 0 since upwelling/downwelling is a diverging quantity)
%% ==========================================
Wlims = zeros(ndep, 2);

for id = 1:ndep
    w_samples = [];
    for it = 1:5:nt   % subsample every 5th step for speed; use 1:nt for exactness
        Wd = squeeze(ncread(files{id}, var_name, [1 1 1 it], [Inf Inf Inf 1]))';
        Wd(land_masks{id}) = NaN;
        w_samples = [w_samples; Wd(:)];
    end
    w_samples(isnan(w_samples)) = [];
    wlim = prctile(abs(w_samples), 99);
    Wlims(id, :) = [-wlim, wlim];
end

disp('Depth-specific color limits (m/s):')
for id = 1:ndep
    fprintf('  %d m: [%.4g, %.4g]\n', depths(id), Wlims(id,1), Wlims(id,2));
end

%% ---- Video setup ----
fname = 'Upwelling_visualization.mp4';

if exist(fname, 'file')
    delete(fname)
end

vid = VideoWriter(fname, 'MPEG-4');
vid.FrameRate = 10;
vid.Quality   = 95;
open(vid);

fig = figure('Color', 'k', 'Position', [100 100 1600 500]);

%% ==========================================
% TIME LOOP
%% ==========================================
for it = 1:nt

    clf

    for id = 1:ndep
        W = squeeze(ncread(files{id}, var_name, [1 1 1 it], [Inf Inf Inf 1]))';
        W(land_masks{id}) = NaN;

        ax = subplot(1, ndep, id);
        pcolor(x, y, W); shading flat
        colormap(ax, cmocean('balance')); caxis(ax, Wlims(id, :))
        axis equal tight
        title(sprintf('%d m Upwelling Velocity', depths(id)), 'Color', 'w')
        xlabel('x (km)', 'Color', 'w'); ylabel('y (km)', 'Color', 'w')
        cb = colorbar(ax); ylabel(cb, 'w (m s^{-1})')
        set(ax, 'XColor', 'w', 'YColor', 'w', 'Color', 'k')
    end

    sg = sgtitle(sprintf('Time = %.2f days', time(it)/86400), ...
        'FontSize', 16, 'FontWeight', 'bold');
    sg.Color = 'w';

    drawnow limitrate
    writeVideo(vid, getframe(fig));

end

close(vid)
%% ==========================================
disp('Upwelling movie saved successfully.')