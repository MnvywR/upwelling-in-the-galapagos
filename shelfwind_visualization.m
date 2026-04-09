%% ==========================================
% Shelfwind Visualization Movie
% T, S, KE, and Vorticity
% Surface + Mid-y Sections
%% ==========================================

clear; close all;
addpath('cmocean')

file_top  = 'top_real.nc';
file_midy = 'midy_real.nc';

%% ---- Load coordinates ----
x    = ncread(file_top,  'x_caa')/1000;
y    = ncread(file_top,  'y_aca')/1000;
time = ncread(file_top,  'time');
x2   = ncread(file_midy, 'x_caa')/1000;
z2   = ncread(file_midy, 'z_aac');

nt = length(time);

%% ---- Color limits ----
Tlims    = [6 28];
Slims    = [33 35.5];
KElims   = [0 0.5];
VORTlims = [-3e-6 3e-6];

%% ---- Video setup ----
vid = VideoWriter('shelfwind_full_visualization.mp4','MPEG-4');
vid.FrameRate = 10;
vid.Quality   = 95;
open(vid);

fig = figure('Color','k','Position',[100 100 1400 1000]);

%% ==========================================
% TIME LOOP
%% ==========================================
for it = 1:nt

    %% ---- Load data ----
    Tsurf    = squeeze(ncread(file_top,  'T',           [1 1 1 it],[Inf Inf Inf 1]))';
    Ssurf    = squeeze(ncread(file_top,  'S',           [1 1 1 it],[Inf Inf Inf 1]))';
    KEsurf   = squeeze(ncread(file_top,  'KE_total',    [1 1 1 it],[Inf Inf Inf 1]))';
    VORTsurf = squeeze(ncread(file_top,  'vorticity_z', [1 1 1 it],[Inf Inf Inf 1]))';

    Tsec     = squeeze(ncread(file_midy, 'T',           [1 1 1 it],[Inf Inf Inf 1]))';
    Ssec     = squeeze(ncread(file_midy, 'S',           [1 1 1 it],[Inf Inf Inf 1]))';
    KEsec    = squeeze(ncread(file_midy, 'KE_total',    [1 1 1 it],[Inf Inf Inf 1]))';
    VORTsec  = squeeze(ncread(file_midy, 'vorticity_z', [1 1 1 it],[Inf Inf Inf 1]))';

    %% ---- NaN mask from T — crop all fields to T size first, then stamp NaNs ----
    [ny_t, nx_t] = size(Tsurf);
    [nz_t, nx_t2] = size(Tsec);

    % Crop surface fields to match T size
    Ssurf    = Ssurf(1:ny_t,   1:nx_t);
    KEsurf   = KEsurf(1:ny_t,  1:nx_t);
    VORTsurf = VORTsurf(1:ny_t, 1:nx_t);

    % Crop section fields to match Tsec size
    Ssec  = Ssec(1:nz_t,  1:nx_t2);
    KEsec = KEsec(1:nz_t, 1:nx_t2);
    VORTsec = VORTsec(1:nz_t, 1:nx_t2);

    % Stamp NaNs from T onto everything
    Ssurf(isnan(Tsurf))    = NaN;
    KEsurf(isnan(Tsurf))   = NaN;
    VORTsurf(isnan(Tsurf)) = NaN;

    Ssec(isnan(Tsec))    = NaN;
    KEsec(isnan(Tsec))   = NaN;
    VORTsec(isnan(Tsec)) = NaN;

    clf

    %% ===============================
    % TEMPERATURE
    %% ===============================
    ax1 = subplot(4,2,1);
    pcolor(x, y, Tsurf); shading flat          % flat = no color bleeding into NaNs
    colormap(ax1, cmocean('thermal')); caxis(Tlims)
    axis equal tight
    title('Surface Temperature','Color','w')
    xlabel('x (km)','Color','w'); ylabel('y (km)','Color','w')
    cb = colorbar; ylabel(cb,'Temperature (°C)')
    set(ax1,'XColor','w','YColor','w','Color','k')

    ax2 = subplot(4,2,2);
    pcolor(x2, z2, Tsec); shading flat
    colormap(ax2, cmocean('thermal')); caxis(Tlims)
    axis tight
    title('Mid-y Temperature','Color','w')
    xlabel('x (km)','Color','w'); ylabel('Depth (m)','Color','w')
    cb = colorbar; ylabel(cb,'Temperature (°C)')
    set(ax2,'XColor','w','YColor','w','Color','k')

    %% ===============================
    % SALINITY
    %% ===============================
    ax3 = subplot(4,2,3);
    pcolor(x, y, Ssurf); shading flat
    colormap(ax3, cmocean('haline')); caxis(Slims)
    axis equal tight
    title('Surface Salinity','Color','w')
    xlabel('x (km)','Color','w'); ylabel('y (km)','Color','w')
    cb = colorbar; ylabel(cb,'Salinity (psu)')
    set(ax3,'XColor','w','YColor','w','Color','k')

    ax4 = subplot(4,2,4);
    pcolor(x2, z2, Ssec); shading flat
    colormap(ax4, cmocean('haline')); caxis(Slims)
    axis tight
    title('Mid-y Salinity','Color','w')
    xlabel('x (km)','Color','w'); ylabel('Depth (m)','Color','w')
    cb = colorbar; ylabel(cb,'Salinity (psu)')
    set(ax4,'XColor','w','YColor','w','Color','k')

    %% ===============================
    % KINETIC ENERGY
    %% ===============================
    ax5 = subplot(4,2,5);
    pcolor(x, y, KEsurf); shading flat
    colormap(ax5, cmocean('speed')); caxis(KElims)
    axis equal tight
    title('Surface Kinetic Energy','Color','w')
    xlabel('x (km)','Color','w'); ylabel('y (km)','Color','w')
    cb = colorbar; ylabel(cb,'Kinetic Energy (J kg^{-1})')
    set(ax5,'XColor','w','YColor','w','Color','k')

    ax6 = subplot(4,2,6);
    pcolor(x2, z2, KEsec); shading flat
    colormap(ax6, cmocean('speed')); caxis(KElims)
    axis tight
    title('Mid-y Kinetic Energy','Color','w')
    xlabel('x (km)','Color','w'); ylabel('Depth (m)','Color','w')
    cb = colorbar; ylabel(cb,'Kinetic Energy (J kg^{-1})')
    set(ax6,'XColor','w','YColor','w','Color','k')

    %% ===============================
    % VORTICITY
    %% ===============================
    ax7 = subplot(4,2,7);
    pcolor(x, y, VORTsurf); shading flat
    colormap(ax7, cmocean('balance')); caxis(VORTlims)
    axis equal tight
    title('Surface Vorticity','Color','w')
    xlabel('x (km)','Color','w'); ylabel('y (km)','Color','w')
    cb = colorbar; ylabel(cb,'Vorticity (s^{-1})')
    set(ax7,'XColor','w','YColor','w','Color','k')

    ax8 = subplot(4,2,8);
    pcolor(x2, z2, VORTsec); shading flat
    colormap(ax8, cmocean('balance')); caxis(VORTlims)
    axis tight
    title('Mid-y Vorticity','Color','w')
    xlabel('x (km)','Color','w'); ylabel('Depth (m)','Color','w')
    cb = colorbar; ylabel(cb,'Vorticity (s^{-1})')
    set(ax8,'XColor','w','YColor','w','Color','k')

    %% ---- Title ----
    sg = sgtitle(sprintf('Time = %.2f days', time(it)/86400), ...
        'FontSize',16,'FontWeight','bold');
    sg.Color = 'w';

    drawnow limitrate
    writeVideo(vid, getframe(fig));

end

close(vid)
disp('Movie saved successfully.')