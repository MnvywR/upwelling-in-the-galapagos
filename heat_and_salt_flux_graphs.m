%% ==========================================
% Graph creation for the heat and salt fluxes
% Might also include some sort of visual of the physical density changes
%% ==========================================

%% ==========================================
%%CONTROL SWITCH

bathymetry_mode = 2; % 0 = no_bathymetry, 1 = gaussian_bathymetry, 2 = real_bathymetry

wind = 0; % 0 = nowind, 1 = wind

beta = 1; % 0 = no beta, 1 = beta

%% Tags
if bathymetry_mode == 0
    bathy_tag = 'no_bathymetry'; bathy_folder = 'galapagos_netcdf_no_bathymetry';
elseif bathymetry_mode == 1
    bathy_tag = 'gaussian_bathymetry'; bathy_folder = 'galapagos_netcdf_gaussian_bathymetry';
elseif bathymetry_mode == 2
    bathy_tag = 'real_bathymetry'; bathy_folder = 'galapagos_netcdf_real_bathymetry';
end
if wind == 0; wind_tag = 'no_wind'; else; wind_tag = 'wind'; end
if beta == 0; beta_tag = 'no_beta'; else; beta_tag = 'beta'; end

%% Load spatial flux fields and integrate over x and y manually

files  = {file_75m, file_150m, file_225m};
labels = {'z = −75 m', 'z = −150 m', 'z = −225 m'};

% Grid spacing — read from file (same for all three)
dx = double(ncread(file_75m, 'Δx_caa'));   % (Nx,1) in metres
dy = double(ncread(file_75m, 'Δy_aca'));   % (Ny,1) in metres
dA = dx * dy';                              % (Nx, Ny) area of each cell

iwT = zeros(101, 3);
iwS = zeros(101, 3);

for k = 1:3
    % wT_difference is (Nx, Ny, 1, Nt)
    wT_field = double(ncread(files{k}, 'wT_difference'));  % (50,50,1,101)
    wS_field = double(ncread(files{k}, 'wS_difference'));

    wT_field = squeeze(wT_field);   % (50, 50, 101)
    wS_field = squeeze(wS_field);

    for it = 1:101
        iwT(it, k) = sum(wT_field(:,:,it) .* dA, 'all');
        iwS(it, k) = sum(wS_field(:,:,it) .* dA, 'all');
    end
end

time = ncread(file_75m, 'time') / 86400;
%% Load time axis (same across all three files)
time = ncread(file_75m, 'time') / 86400;  % seconds → days

%% Load volume-integrated fluxes — unicode variable names read fine with ncread
iwT_75  = ncread(file_75m,  '∫wT_difference_up');
iwT_150 = ncread(file_150m, '∫wT_difference_up');
iwT_225 = ncread(file_225m, '∫wT_difference_up');

iwS_75  = ncread(file_75m,  '∫wS_difference_up');
iwS_150 = ncread(file_150m, '∫wS_difference_up');
iwS_225 = ncread(file_225m, '∫wS_difference_up');

%% Plot
figure('Position', [100 100 900 600])

subplot(2,1,1)
colors = {'r','b','g'};
for k = 1:3
    plot(time, iwT(:,k), colors{k}, 'LineWidth', 1.8, 'DisplayName', labels{k}); hold on
end
yline(0,'k--','LineWidth',0.8,'HandleVisibility','off'); hold off
xlabel('Time (days)'); ylabel('\intw''T'' dV  (m^3 s^{-1} °C)')
title('Volume-integrated upward heat flux anomaly'); legend; grid on

subplot(2,1,2)
for k = 1:3
    plot(time, iwS(:,k), colors{k}, 'LineWidth', 1.8, 'DisplayName', labels{k}); hold on
end
yline(0,'k--','LineWidth',0.8,'HandleVisibility','off'); hold off
xlabel('Time (days)'); ylabel('\intw''S'' dV  (m^3 s^{-1} psu)')
title('Volume-integrated upward salt flux anomaly'); legend; grid on