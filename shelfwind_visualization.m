%% ==========================================
% Shelfwind Visualization Movie
% T, S, KE, and Vorticity
% Surface + Mid-y Sections
%% ==========================================

%% ==========================================
%%CONTROL SWITCH

bathymetry_mode = 2;  % 0 = no_bathymetry, 1 = gaussian_bathymetry, 2 = real_bathymetry

wind = 0;             % 0 = nowind, 1 = wind

beta = 1;             % 0 = no beta plane, 1 = beta plane

%% ==========================================

if bathymetry_mode == 0
    bathy_tag    = 'no_bathymetry';
    bathy_folder = 'galapagos_netcdf_no_bathymetry';
elseif bathymetry_mode == 1
    bathy_tag    = 'gaussian_bathymetry';
    bathy_folder = 'galapagos_netcdf_gaussian_bathymetry';
elseif bathymetry_mode == 2
    bathy_tag    = 'real_bathymetry';
    bathy_folder = 'galapagos_netcdf_real_bathymetry';
end

if wind == 0
    wind_tag = 'no_wind';
elseif wind == 1
    wind_tag = 'wind';
end

if beta == 0
    beta_tag = 'no_beta';
elseif beta == 1
    beta_tag = 'beta';
end

file_top  = fullfile(bathy_folder, sprintf('top_%s_%s_%s_GPU.nc',  bathy_tag, wind_tag, beta_tag));
file_midy = fullfile(bathy_folder, sprintf('midy_%s_%s_%s_GPU.nc', bathy_tag, wind_tag, beta_tag));

%% ---- Load coordinates ----
x    = ncread(file_top,  'x_caa')/1000;
y    = ncread(file_top,  'y_aca')/1000;
time = ncread(file_top,  'time');
x2   = ncread(file_midy, 'x_caa')/1000;
z2   = ncread(file_midy, 'z_aac');

nt = length(time);

%{

%% ---- Auto-scan for robust surface color limits (run once, before the main loop) ----
T_surf_samples    = [];
S_surf_samples    = [];
KE_surf_samples   = [];
vort_surf_samples = [];

for it = 1:5:nt   % subsample every 5th step for speed; use 1:nt for exactness
    Ts = squeeze(ncread(file_top, 'T',           [1 1 1 it],[Inf Inf Inf 1]));
    Ss = squeeze(ncread(file_top, 'S',           [1 1 1 it],[Inf Inf Inf 1]));
    Ks = squeeze(ncread(file_top, 'KE_total',    [1 1 1 it],[Inf Inf Inf 1]));
    Vs = squeeze(ncread(file_top, 'vorticity_z', [1 1 1 it],[Inf Inf Inf 1]));

    % Mask out land using T (same logic as your main loop)
    land = isnan(Ts) | (Ts <= 0);
    Ts(land) = NaN; Ss(land) = NaN; Ks(land) = NaN; Vs(land) = NaN;

    T_surf_samples    = [T_surf_samples;    Ts(:)];
    S_surf_samples    = [S_surf_samples;    Ss(:)];
    KE_surf_samples   = [KE_surf_samples;   Ks(:)];
    vort_surf_samples = [vort_surf_samples; Vs(:)];
end

T_surf_samples(isnan(T_surf_samples))       = [];
S_surf_samples(isnan(S_surf_samples))       = [];
KE_surf_samples(isnan(KE_surf_samples))     = [];
vort_surf_samples(isnan(vort_surf_samples)) = [];

%% ---- Temperature & Salinity: use 1st/99th percentile (not symmetric, not diverging) ----
Tlims_surf = [prctile(T_surf_samples, 1), prctile(T_surf_samples, 99)];
Slims_surf = [prctile(S_surf_samples, 1), prctile(S_surf_samples, 99)];

%% ---- KE: physically bounded at 0, so just take the upper 99th percentile ----
KElims_surf = [0, prctile(KE_surf_samples, 99)];

%% ---- Vorticity: symmetric about 0 (diverging colormap) ----
vlim_surf = prctile(abs(vort_surf_samples), 99);
VORTlims_surf = [-vlim_surf, vlim_surf];

disp('Surface color limits:')
disp(['Tlims_surf    = [', num2str(Tlims_surf), ']'])
disp(['Slims_surf    = [', num2str(Slims_surf), ']'])
disp(['KElims_surf   = [', num2str(KElims_surf), ']'])
disp(['VORTlims_surf = [', num2str(VORTlims_surf), ']'])


%}


%% ---- Color limits (INDEPENDENT for surface xy vs section xz) ----
% Surface (xy) panels

Tlims_surf    = [20.6, 21.8];
Slims_surf    = [34.965, 34.995];
KElims_surf   = [0, 0.0146];
VORTlims_surf = [-5.44e-6, 5.44e-6];

% Mid-y section (xz) panels
Tlims_sec    = [6 28];
Slims_sec    = [34 35.5];
KElims_sec   = [0 0.05];
VORTlims_sec = [-2e-6 2e-6];

%% ---- Video setup ----

fname = 'Galapagos_full_visualization.mp4';

if exist(fname,'file')
    delete(fname)
end

vid = VideoWriter(fname,'MPEG-4');
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
    %% ---- Build land masks ----
    land_surf = isnan(Tsurf) | (Tsurf <= 0);
    land_sec  = isnan(Tsec)  | (Tsec <= 0);

    %% ---- Apply masks to ALL fields ----
    Tsurf(land_surf)     = NaN;
    Ssurf(land_surf)     = NaN;
    KEsurf(land_surf)    = NaN;
    VORTsurf(land_surf)  = NaN;

    Tsec(land_sec)       = NaN;
    Ssec(land_sec)       = NaN;
    KEsec(land_sec)      = NaN;
    VORTsec(land_sec)    = NaN;

    clf

    %% ===============================
    % TEMPERATURE
    %% ===============================
    ax1 = subplot(4,2,1);
    pcolor(x, y, Tsurf); shading flat          % flat = no color bleeding into NaNs
    colormap(ax1, cmocean('thermal')); caxis(ax1, Tlims_surf)
    axis equal tight
    title('Surface Temperature','Color','w')
    xlabel('x (km)','Color','w'); ylabel('y (km)','Color','w')
    cb = colorbar(ax1); ylabel(cb,'Temperature (°C)')
    set(ax1,'XColor','w','YColor','w','Color','k')

    ax2 = subplot(4,2,2);
    pcolor(x2, z2, Tsec); shading flat
    colormap(ax2, cmocean('thermal')); caxis(ax2, Tlims_sec)
    axis tight
    title('Mid-y Temperature','Color','w')
    xlabel('x (km)','Color','w'); ylabel('Depth (m)','Color','w')
    cb = colorbar(ax2); ylabel(cb,'Temperature (°C)')
    set(ax2,'XColor','w','YColor','w','Color','k')

    %% ===============================
    % SALINITY
    %% ===============================
    ax3 = subplot(4,2,3);
    pcolor(x, y, Ssurf); shading flat
    colormap(ax3, cmocean('haline')); caxis(ax3, Slims_surf)
    axis equal tight
    title('Surface Salinity','Color','w')
    xlabel('x (km)','Color','w'); ylabel('y (km)','Color','w')
    cb = colorbar(ax3); ylabel(cb,'Salinity (psu)')
    set(ax3,'XColor','w','YColor','w','Color','k')

    ax4 = subplot(4,2,4);
    pcolor(x2, z2, Ssec); shading flat
    colormap(ax4, cmocean('haline')); caxis(ax4, Slims_sec)
    axis tight
    title('Mid-y Salinity','Color','w')
    xlabel('x (km)','Color','w'); ylabel('Depth (m)','Color','w')
    cb = colorbar(ax4); ylabel(cb,'Salinity (psu)')
    set(ax4,'XColor','w','YColor','w','Color','k')

    %% ===============================
    % KINETIC ENERGY
    %% ===============================
    ax5 = subplot(4,2,5);
    pcolor(x, y, KEsurf); shading flat
    colormap(ax5, cmocean('speed')); caxis(ax5, KElims_surf)
    axis equal tight
    title('Surface Kinetic Energy','Color','w')
    xlabel('x (km)','Color','w'); ylabel('y (km)','Color','w')
    cb = colorbar(ax5); ylabel(cb,'Kinetic Energy (J kg^{-1})')
    set(ax5,'XColor','w','YColor','w','Color','k')

    ax6 = subplot(4,2,6);
    pcolor(x2, z2, KEsec); shading flat
    colormap(ax6, cmocean('speed')); caxis(ax6, KElims_sec)
    axis tight
    title('Mid-y Kinetic Energy','Color','w')
    xlabel('x (km)','Color','w'); ylabel('Depth (m)','Color','w')
    cb = colorbar(ax6); ylabel(cb,'Kinetic Energy (J kg^{-1})')
    set(ax6,'XColor','w','YColor','w','Color','k')

    %% ===============================
    % VORTICITY
    %% ===============================
    ax7 = subplot(4,2,7);
    pcolor(x, y, VORTsurf); shading flat
    colormap(ax7, cmocean('balance')); caxis(ax7, VORTlims_surf)
    axis equal tight
    title('Surface Vorticity','Color','w')
    xlabel('x (km)','Color','w'); ylabel('y (km)','Color','w')
    cb = colorbar(ax7); ylabel(cb,'Vorticity (s^{-1})')
    set(ax7,'XColor','w','YColor','w','Color','k')

    ax8 = subplot(4,2,8);
    pcolor(x2, z2, VORTsec); shading flat
    colormap(ax8, cmocean('balance')); caxis(ax8, VORTlims_sec)
    axis tight
    title('Mid-y Vorticity','Color','w')
    xlabel('x (km)','Color','w'); ylabel('Depth (m)','Color','w')
    cb = colorbar(ax8); ylabel(cb,'Vorticity (s^{-1})')
    set(ax8,'XColor','w','YColor','w','Color','k')

    %% ---- Title ----
    sg = sgtitle(sprintf('Time = %.2f days', time(it)/86400), ...
        'FontSize',16,'FontWeight','bold');
    sg.Color = 'w';

    drawnow limitrate
    writeVideo(vid, getframe(fig));

end

close(vid)
%% ==========================================
disp('Movie saved successfully.')
