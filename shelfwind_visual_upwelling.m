%% ==========================================
% Upwelling at 75m — Contour + Quiver Style
%% ==========================================
clear; close all;

file = 'upwelling_75m.nc';

%% ---- Load coordinates ----
x    = ncread(file, 'x_caa') / 1000;
y    = ncread(file, 'y_aca') / 1000;
time = ncread(file, 'time');
nt   = length(time);

%% ---- Color limits ----
Wlims    = [-5e-4  5e-4];
Tlims    = [10 25];
VORTlims = [-3e-5 3e-5];
Slims    = [33 35.5];

%% ---- Quiver subsample rate ----
qskip = 3;

%% ---- Video setup ----
vid = VideoWriter('upwelling_75m.mp4', 'MPEG-4');
vid.FrameRate = 8;
vid.Quality   = 95;
open(vid);

fig = figure('Color','w', 'Position', [100 100 1200 900]);  % 2x2 needs taller figure

%% ==========================================
% TIME LOOP
%% ==========================================
for it = 1:nt

    %% ---- Load fields ----
    u    = squeeze(ncread(file, 'u',           [1 1 1 it], [Inf Inf Inf 1]))';
    v    = squeeze(ncread(file, 'v',           [1 1 1 it], [Inf Inf Inf 1]))';
    w    = squeeze(ncread(file, 'w',           [1 1 1 it], [Inf Inf Inf 1]))';
    T    = squeeze(ncread(file, 'T',           [1 1 1 it], [Inf Inf Inf 1]))';
    S    = squeeze(ncread(file, 'S',           [1 1 1 it], [Inf Inf Inf 1]))';
    vort = squeeze(ncread(file, 'vorticity_z', [1 1 1 it], [Inf Inf Inf 1]))';

    %% ---- Crop + NaN mask ----
    [ny, nx] = size(T);
    u    = u(1:ny, 1:nx);
    v    = v(1:ny, 1:nx);
    w    = w(1:ny, 1:nx);
    S    = S(1:ny, 1:nx);
    vort = vort(1:ny, 1:nx);

    mask = isnan(T);
    u(mask) = NaN;  v(mask) = NaN;
    w(mask) = NaN;  vort(mask) = NaN;
    S(mask) = NaN;

    %% ---- Quiver subsampling ----
    xq = x(1:qskip:end);
    yq = y(1:qskip:end);
    uq = u(1:qskip:end, 1:qskip:end);
    vq = v(1:qskip:end, 1:qskip:end);
    [Xq, Yq] = meshgrid(xq, yq);

    clf

    %% ----------------------------
    % Panel 1: Vertical velocity w
    %% ----------------------------
    ax1 = subplot(2, 2, 1);
    hold on
    contourf(x, y, w, 30, 'LineColor', 'none');
    colormap(ax1, bluewhitered());
    clim(Wlims)
    cb = colorbar; ylabel(cb, 'w  (m s^{-1})', 'FontSize', 10)
    %quiver(Xq, Yq, uq, vq, 1.5, 'k', 'LineWidth', 0.6, 'MaxHeadSize', 0.4)
    %contour(x, y, w, [0 0], 'k-', 'LineWidth', 1.5)
    axis equal tight
    title('Vertical Velocity  w  (+ = upwelling)', 'FontSize', 11, 'FontWeight', 'bold')
    xlabel('x  (km)');  ylabel('y  (km)')
    set(ax1, 'FontSize', 10, 'Box', 'on')
    hold off

    %% ----------------------------
    % Panel 2: Temperature + w contours
    %% ----------------------------
    ax2 = subplot(2, 2, 2);
    hold on
    contourf(x, y, T, 25, 'LineColor', 'none');
    colormap(ax2, cmocean('thermal'));
    clim(Tlims)
    cb = colorbar; ylabel(cb, 'T  (°C)', 'FontSize', 10)
    %contour(x, y, w, [-5e-4 -2e-4 -1e-4], 'b--', 'LineWidth', 1.0)
    %contour(x, y, w, [ 1e-4  2e-4  5e-4], 'r-',  'LineWidth', 1.0)
    axis equal tight
    title('Temperature  +  w contours', 'FontSize', 11, 'FontWeight', 'bold')
    xlabel('x  (km)');  ylabel('y  (km)')
    set(ax2, 'FontSize', 10, 'Box', 'on')
    hold off

    %% ----------------------------
    % Panel 3: Salinity + w contours
    %% ----------------------------
    ax3 = subplot(2, 2, 3);
    hold on
    contourf(x, y, S, 25, 'LineColor', 'none');
    colormap(ax3, cmocean('haline'));
    clim(Slims)
    cb = colorbar; ylabel(cb, 'S  (psu)', 'FontSize', 10)
    % Same w contours as T panel — upwelling brings fresher water up
    %contour(x, y, w, [-5e-4 -2e-4 -1e-4], 'b--', 'LineWidth', 1.0)
    %contour(x, y, w, [ 1e-4  2e-4  5e-4], 'r-',  'LineWidth', 1.0)
    axis equal tight
    title('Salinity  +  w contours', 'FontSize', 11, 'FontWeight', 'bold')
    xlabel('x  (km)');  ylabel('y  (km)')
    set(ax3, 'FontSize', 10, 'Box', 'on')
    hold off

    %% ----------------------------
    % Panel 4: Vorticity
    %% ----------------------------
    ax4 = subplot(2, 2, 4);
    hold on
    contourf(x, y, vort, 30, 'LineColor', 'none');
    colormap(ax4, bluewhitered());
    clim(VORTlims)
    cb = colorbar; ylabel(cb, '\omega_z  (s^{-1})', 'FontSize', 10)
    %quiver(Xq, Yq, uq, vq, 1.5, 'k', 'LineWidth', 0.6, 'MaxHeadSize', 0.4)
    axis equal tight
    title('Vertical Vorticity  \omega_z', 'FontSize', 11, 'FontWeight', 'bold')
    xlabel('x  (km)');  ylabel('y  (km)')
    set(ax4, 'FontSize', 10, 'Box', 'on')
    hold off

    sgtitle(sprintf('Depth = 75 m  |  Time = %.2f days', time(it)/86400), ...
        'FontSize', 14, 'FontWeight', 'bold')

    drawnow limitrate
    writeVideo(vid, getframe(fig));
end

close(vid)
disp('Saved upwelling_75m.mp4')


function cmap = bluewhitered(n)
    if nargin < 1; n = 256; end
    half = floor(n/2);
    r1 = linspace(0.1, 1, half)';
    g1 = linspace(0.3, 1, half)';
    b1 = ones(half, 1);
    r2 = ones(n-half, 1);
    g2 = linspace(1, 0.1, n-half)';
    b2 = linspace(1, 0.1, n-half)';
    cmap = [r1 g1 b1; r2 g2 b2];
end