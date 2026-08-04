function shelfwind_visualization(outdir)
%% ==========================================
% Shelfwind Visualization Movie
% T, S, KE, and Vorticity
% Surface + Mid-y Sections
%
% outdir: the folder produced by one run_simulation() call from the Julia
% driver. Expects top_.nc and midy.nc inside it, and writes the resulting
% movie back into that same folder.
%% ==========================================

file_top  = fullfile(outdir, 'top_.nc');
file_midy = fullfile(outdir, 'midy.nc');

%% ---- Load coordinates ----
x    = ncread(file_top,  'x_caa')/1000;
y    = ncread(file_top,  'y_aca')/1000;
time = ncread(file_top,  'time');
x2   = ncread(file_midy, 'x_caa')/1000;
z2   = ncread(file_midy, 'z_aac');

nt = length(time);

%% ---- Auto-scan for robust color limits (surface + section) ----
% This batch now runs 7 different configs (bathymetry on/off, beta on/off,
% different EUC values) through the same script, so fixed hardcoded caxis
% limits tuned for one run won't generalize — scan each run's own data
% instead. Subsampled to ~20 timesteps for speed rather than every frame.

T_surf_samples = []; S_surf_samples = []; KE_surf_samples = []; vort_surf_samples = [];
T_sec_samples  = []; S_sec_samples  = []; KE_sec_samples  = []; vort_sec_samples  = [];

scan_stride = max(1, floor(nt/20));

for it = 1:scan_stride:nt
    Ts = squeeze(ncread(file_top, 'T',           [1 1 1 it],[Inf Inf Inf 1]));
    Ss = squeeze(ncread(file_top, 'S',           [1 1 1 it],[Inf Inf Inf 1]));
    Ks = squeeze(ncread(file_top, 'KE_total',    [1 1 1 it],[Inf Inf Inf 1]));
    Vs = squeeze(ncread(file_top, 'vorticity_z', [1 1 1 it],[Inf Inf Inf 1]));
    land = isnan(Ts) | (Ts <= 0);
    Ts(land) = NaN; Ss(land) = NaN; Ks(land) = NaN; Vs(land) = NaN;
    T_surf_samples    = [T_surf_samples;    Ts(:)];
    S_surf_samples    = [S_surf_samples;    Ss(:)];
    KE_surf_samples   = [KE_surf_samples;   Ks(:)];
    vort_surf_samples = [vort_surf_samples; Vs(:)];

    Tc = squeeze(ncread(file_midy, 'T',           [1 1 1 it],[Inf Inf Inf 1]));
    Sc = squeeze(ncread(file_midy, 'S',           [1 1 1 it],[Inf Inf Inf 1]));
    Kc = squeeze(ncread(file_midy, 'KE_total',    [1 1 1 it],[Inf Inf Inf 1]));
    Vc = squeeze(ncread(file_midy, 'vorticity_z', [1 1 1 it],[Inf Inf Inf 1]));
    land_c = isnan(Tc) | (Tc <= 0);
    Tc(land_c) = NaN; Sc(land_c) = NaN; Kc(land_c) = NaN; Vc(land_c) = NaN;
    T_sec_samples     = [T_sec_samples;     Tc(:)];
    S_sec_samples     = [S_sec_samples;     Sc(:)];
    KE_sec_samples    = [KE_sec_samples;    Kc(:)];
    vort_sec_samples  = [vort_sec_samples;  Vc(:)];
end

T_surf_samples(isnan(T_surf_samples))       = [];
S_surf_samples(isnan(S_surf_samples))       = [];
KE_surf_samples(isnan(KE_surf_samples))     = [];
vort_surf_samples(isnan(vort_surf_samples)) = [];
T_sec_samples(isnan(T_sec_samples))         = [];
S_sec_samples(isnan(S_sec_samples))         = [];
KE_sec_samples(isnan(KE_sec_samples))       = [];
vort_sec_samples(isnan(vort_sec_samples))   = [];

Tlims_surf    = [prctile(T_surf_samples, 1), prctile(T_surf_samples, 99)];
Slims_surf    = [prctile(S_surf_samples, 1), prctile(S_surf_samples, 99)];
KElims_surf   = [0, prctile(KE_surf_samples, 99)];
vlim_surf     = prctile(abs(vort_surf_samples), 99);
VORTlims_surf = [-vlim_surf, vlim_surf];

Tlims_sec    = [prctile(T_sec_samples, 1), prctile(T_sec_samples, 99)];
Slims_sec    = [prctile(S_sec_samples, 1), prctile(S_sec_samples, 99)];
KElims_sec   = [0, prctile(KE_sec_samples, 99)];
vlim_sec     = prctile(abs(vort_sec_samples), 99);
VORTlims_sec = [-vlim_sec, vlim_sec];

%% ---- Video setup ----
fname = fullfile(outdir, 'Galapagos_full_visualization.mp4');

if exist(fname,'file')
    delete(fname)
end

vid = VideoWriter(fname,'MPEG-4');
vid.FrameRate = 10;
vid.Quality   = 95;
open(vid);

% Visible off: this runs headless via `matlab -batch`, no display attached.
fig = figure('Color','k','Position',[100 100 1400 1000], 'Visible','off');

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
close(fig)
fprintf('Movie saved: %s\n', fname);

end