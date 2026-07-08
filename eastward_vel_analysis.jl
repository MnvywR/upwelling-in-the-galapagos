    using Pkg
    using NCDatasets 
    using Plots
    using FFTW
    using Dates
    using Statistics
    using Glob
    

    #---------------------
    # Target location (shared by daily and monthly analysis)
    #---------------------
    target_depth     = 77.85385f0       # ~78m
    target_longitude = -94.99999f0      # ~95W
    target_latitude  =  0.33333334f0   # ~0.33N
    #---------------------
    #importing daily data
    #---------------------

    #IMPORTING DATA SETS HERE:

    daily_data_nc = NCDataset("eastward_vel_daily_data/daily_eastward_sea_velocity_2000_to_2026.nc")  # update filename to match yours

    daily_depth     = Float64.(daily_data_nc["depth"][:])
    daily_longitude = Float64.(daily_data_nc["longitude"][:])
    daily_latitude  = Float64.(daily_data_nc["latitude"][:])
    daily_time      = daily_data_nc["time"][:]

    #LOADING IN DATA FROM THE DAILY DATASET:

    # Find the indices of the nearest longitude, latitude, and depth, which are index 21 for depth, index 47 for lat, index 37 for longitude.
    i_lon = argmin(abs.(daily_longitude .- target_longitude))
    i_lat = argmin(abs.(daily_latitude  .- target_latitude))
    i_dep = argmin(abs.(daily_depth     .- target_depth))

    # Filter to first year (2018) — adjust as needed
    start_date   = DateTime(2000, 1, 1)
    end_date     = DateTime(2026, 1, 1)
    time_mask    = (daily_time .>= start_date) .& (daily_time .<= end_date)
    time_indices = findall(time_mask)


    # u dims: (longitude × latitude × depth × time) 
    new_daily_data = Float64.(coalesce.(daily_data_nc["uo"][i_lon, i_lat, i_dep, time_indices], NaN))
    
    close(daily_data_nc)

    # Replace any remaining NaN with the mean
    valid_mean = mean(filter(!isnan, new_daily_data))
    u_clean    = map(v -> isnan(v) ? valid_mean : v, new_daily_data)
    
    #---------------------
    # Plot raw velocity
    #---------------------
    plot_daily1 = Plots.plot(
    daily_time[time_indices], u_clean,
    title  = "Eastward Sea Velocity at ~78m, 95°W, 0.33°N (2017–2021)",
    xlabel = "Date",
    ylabel = "Velocity (m/s)",
    label  = "u",
    lw     = 1.5,
    size   = (900, 500)
    )
   
    #-----------------------
    #FFT of daily data
    #-----------------------
    N_days = length(u_clean) #How many total days
    1/N_days #Frequency resolution
    Sampling_rate = 1.0 # 1 sample per day
    Frequencies = (0:N_days-1) * (Sampling_rate / N_days) # cycles per day
    
    #Remove the mean
    u_new = u_clean .- mean(u_clean)

    hann = 0.5 .- 0.5*cos.(2π*(0:N_days-1)/(N_days-1)) #Hann window to reduce spectral leakage
    u_windowed = u_new .* hann

    #Perform FFT
    u_fft = fft(u_windowed)
    half_range = 2:div(N_days, 2) # Array from 1 to N/2, since it'll be a mirror image
    energy_spectrum = abs.(u_fft[half_range]).^2 #Energy spectrum (magnitude squared)
    period = 1.0 ./ Frequencies[half_range] # Convert frequencies to periods (days)
    
    #---------------------------------
    #PLOTS
    #---------------------------------

    plot_fft_freq =Plots.plot(
        Frequencies, energy_spectrum,
        xlabel = "Frequency (cycles/day)",
        seriestype = :sticks,
        grid = true,
        ylabel = "Energy",
        title  = "Energy Spectrum of Daily Velocity (2000–2026)",
        lw = 1.5,
        label = "Power"
    )
    display(plot_fft_freq)

    plot_fft_freq_notsticks =Plots.plot(
        Frequencies[half_range], energy_spectrum,
        xlabel = "Frequency (cycles/day)",
        grid = true,
        ylabel = "Energy",
        title  = "Energy Spectrum of Daily Velocity (2000–2026)",
        lw = 1.5,
        label = "Power"
    )
    display(plot_fft_freq_notsticks)

    plot_fft_days = Plots.plot(
        period[2:end], energy_spectrum[2:end],
        xlabel = "Period (days)",
        seriestype = :sticks,
        grid = true,
        ylabel = "Energy",
        title  = "Energy Spectrum of Daily Velocity (2000–2026)",
        lw = 1.5,
        label = "Power",
        xticks = 0:100:5000,
        xrotation = 45
    )
    display(plot_fft_days)

    
    #----------------------
    #importing monthly data
    #----------------------

    monthly_data_nc = NCDataset("eastward_sea_velocity.nc")

    monthly_depth     = Float64.(monthly_data_nc["depth"][:])
    monthly_longitude = Float64.(monthly_data_nc["longitude"][:])
    monthly_latitude  = Float64.(monthly_data_nc["latitude"][:])
    monthly_time      = monthly_data_nc["time"][:]

    i_lon = argmin(abs.(monthly_longitude .- target_longitude))
    i_lat = argmin(abs.(monthly_latitude  .- target_latitude))
    i_dep = argmin(abs.(monthly_depth     .- target_depth))
    monthly_time_mask = (monthly_time .>= start_date) .& (monthly_time .<= end_date)
    monthly_indices   = findall(monthly_time_mask)

    new_monthly_data  = monthly_data_nc["uo"][i_lon, i_lat, i_dep, monthly_indices]

    u_monthly_clean = Float64.(coalesce.(new_monthly_data, mean(skipmissing(new_monthly_data))))

    raw_monthly_velocity = Plots.plot(
        monthly_time[monthly_indices], u_monthly_clean,
        title="Eastward Sea Velocity at 75m depth, 95W, 0.33N", 
        xlabel="Time (months)", 
        ylabel="Velocity (m/s)"
        )
    display(raw_monthly_velocity)

    #----------------------
    #FFT of monthly data
    #----------------------

    N_months = length(u_monthly_clean) #How many total days
    1/N_months #Frequency resolution
    Sampling_rate = 1.0 # 1 sample per month
    Frequencies_months = (0:N_months-1) * (Sampling_rate / N_months) # cycles per month
    
    #Remove the mean
    u_monthly_new = u_monthly_clean .- mean(u_monthly_clean)

    hann_monthly = 0.5 .- 0.5*cos.(2π*(0:N_months-1)/(N_months-1)) #Hann window to reduce spectral leakage
    u_monthly_windowed = u_monthly_new .* hann_monthly

    #Perform FFT
    u_monthly_fft = fft(u_monthly_windowed)
    half_range_months = 2:div(N_months, 2) # Array from 1 to N/2, since it'll be a mirror image
    energy_spectrum_months = abs.(u_monthly_fft[half_range_months]).^2 #Energy spectrum (magnitude squared)
    period_months = 1.0 ./ Frequencies_months[half_range_months] # Convert frequencies to periods (months)

    plot_fft_freq_months =Plots.plot(
        Frequencies_months, energy_spectrum_months,
        xlabel = "Frequency (cycles/month)",
        seriestype = :sticks,
        grid = true,
        ylabel = "Energy",
        title  = "Energy Spectrum of Monthly Velocity (2000–2026)",
        lw = 1.5,
        label = "Power"
    )
    display(plot_fft_freq_months)

    plot_fft_freq_notsticks_months =Plots.plot(
        Frequencies_months[half_range_months], energy_spectrum_months,
        xlabel = "Frequency (cycles/month)",
        grid = true,
        ylabel = "Energy",
        title  = "Energy Spectrum of Monthly Velocity (2000–2026)",
        lw = 1.5,
        label = "Power"
    )
    display(plot_fft_freq_notsticks_months)

    plot_fft_months = Plots.plot(
        period_months[1:end], energy_spectrum_months[1:end],
        xlabel = "Period (months)",
        seriestype = :sticks,
        grid = true,
        ylabel = "Energy",
        title  = "Energy Spectrum of Monthly Velocity (2017–2021)",
        lw = 1.5,
        label = "Power",
        xrotation = 45
    )
    display(plot_fft_months)


    #Nyquist limit is maximum frequency that can be accurately recorded (Freq / 2)
    #So for daily -> 2 days
    #For monthly -> 2 months
    #ON ONE GRAPH

    # Convert monthly periods to days (1 month ≈ 30 days)
    period_months_indays = period_months .* 30

    # Normalize both energy spectra so they're on the same scale
    energy_daily_norm   = energy_spectrum ./ maximum(energy_spectrum)
    energy_monthly_norm = energy_spectrum_months ./ maximum(energy_spectrum_months)

    # Plot both together
    plot_combined = Plots.plot(
        period[2:end], energy_daily_norm[2:end],
        xlabel     = "Period (days)",
        seriestype = :sticks,
        ylabel     = "Normalized Energy",
        title      = "Energy Spectrum: Daily vs Monthly (2000–2026)",
        label      = "Daily",
        color      = :steelblue,
        lw         = 1.5,
        xticks     = 0:100:9000,
        xrotation  = 45,
        size       = (900, 700)
    )

    Plots.plot!(plot_combined,
        period_months_indays[2:end], energy_monthly_norm[2:end],
        seriestype = :sticks,
        label      = "Monthly",
        color      = :red,
        lw         = 1.5,
        alpha      = 0.7
    )

    display(plot_combined)

    #------------------------------------------------
    #Pulling peaks from the Daily and Monthly data
    #------------------------------------------------

    #DAILY DATA:

    #The hann window shrinks the amplitude, so I am re-adjusting it
    window_correction = mean(hann)

    top_n = 10  # how many modes to extract

    # Sort by energy, skip DC
    top_idx_daily = sortperm(energy_spectrum, rev=true)[1:top_n]
    top_idx_daily = half_range[top_idx_daily]  # shift back to full array indices

    println("\n=== Top $top_n Daily Modes ===")
    println("Rank | Period (days) | Frequency (cyc/day) | Amplitude (m/s)")
    println("-----|---------------|---------------------|-----------------")
    for (rank, i) in enumerate(top_idx_daily) #Pair up the top index with a rank of the highest modes
        f_i = Frequencies[i] #The frequency
        T_i = 1.0 / f_i #The period
        A_i = (2 * abs(u_fft[i]) / N_days) / window_correction #The amplitude        
        println("  $rank  | $(rpad(round(T_i, digits=1), 13)) | $(rpad(round(f_i, digits=6), 19)) | $(rpad(round(A_i, digits=4), 15))")
    end

    #MONTHLY DATA:

    window_correction_months = mean(hann_monthly)

    top_idx_monthly = sortperm(energy_spectrum_months, rev=true)[1:top_n]
    top_idx_monthly = half_range_months[top_idx_monthly]

    println("\n=== Top $top_n Monthly Modes ===")
    println("Rank | Period (months) | Period (days) | Frequency (cyc/month) | Amplitude (m/s)")
    println("-----|-----------------|---------------|----------------------|-----------------")
    for (rank, i) in enumerate(top_idx_monthly)
        f_i = Frequencies_months[i]
        T_i = 1.0 / f_i
        A_i = (2 * abs(u_monthly_fft[i]) / N_months) / window_correction_months
        println("  $rank  | $(rpad(round(T_i, digits=1), 15)) | $(rpad(round(T_i*30, digits=0), 13)) | $(rpad(round(f_i, digits=6), 20)) | $(rpad(round(A_i, digits=4), 15))")
    end

    #Putting top modes on to the related data to see trend

    t_days = collect(0:length(u_clean)-1)

    u_reconstructed = mean(u_clean) .+
        0.1373 .* cos.(2π .* 0.002737 .* t_days ) #.+
        #0.121  .* cos.(2π .* 0.005475 .* t_days ) .+
        #0.1068 .* cos.(2π .* 0.001895 .* t_days )

    reconstructed_eq_plot = Plots.plot(
        daily_time[time_indices], u_clean,
        label  = "Original data",
        lw     = 1.0,
        alpha  = 0.5,
        color  = :steelblue,
        xlabel = "Date",
        ylabel = "Velocity (m/s)",
        title  = "EUC Velocity + Top 3 Mode Reconstruction (2000–2026)",
        size   = (900, 500)
    )

    Plots.plot!(reconstructed_eq_plot,
        daily_time[time_indices], u_reconstructed,
        label = "Top 3 modes (365d + 183d + 528d)",
        lw    = 2.0,
        color = :red
    )

    display(reconstructed_eq_plot)

    #----------------------------------------------
    #Figuring out phase of top dominant wave modes
    #----------------------------------------------
    #u_fft has both the imaginary and real parts from the FFT
        
    # ---- (b) Physical sanity check: the mean flow at this depth/location
    #          is the Equatorial Undercurrent (EUC), which is known to flow
    #          eastward. If the sign convention is correct AND the data is
    #          physically sound, the long-term mean should be POSITIVE.
    println("\nMean u at 78 m, 95°W, 0.33°N = ", round(mean(u_clean), digits=4), " m/s")
    println("(A positive mean here, combined with a 'positive: east' style")
    println(" attribute above, confirms the data — and therefore anything")
    println(" built from it — is oriented eastward-positive.)")
    
    # ---- (c) Is the reconstruction IN PHASE with the original signal? ----
    # Pearson correlation between the (demeaned) original and the
    # (demeaned) reconstruction. +1 = perfectly in phase, -1 = perfectly
    # inverted (180° out of phase), 0 = unrelated.
    u_recon_demeaned = u_reconstructed .- mean(u_reconstructed)
    phase_corr = cor(u_new, u_recon_demeaned)
    
    println("\nCorrelation between original signal and top-3-mode reconstruction: ",
            round(phase_corr, digits=4))
    if phase_corr > 0
        println("  -> Positive correlation: reconstruction is IN PHASE with the data.")
    else
        println("  -> Negative correlation: reconstruction is 180° OUT OF PHASE.")
        println("     Fix by adding π to each phase term (or flipping the sign of that mode's amplitude).")
    end
    
    # ---- (d) Cross-correlation across small time lags ----
    # Confirms the best agreement really happens at lag = 0 (i.e. there's
    # no leftover time shift hiding underneath a "correct" zero-lag
    # correlation).
    function lagged_corr(a, b, lag)
        n = length(a)
        if lag >= 0
            idx_a = 1:(n-lag)
            idx_b = (1+lag):n
        else
            idx_a = (1-lag):n
            idx_b = 1:(n+lag)
        end
        return cor(a[idx_a], b[idx_b])
    end
    
    max_lag   = 360 # days
    lags      = -max_lag:max_lag
    xcorr_vals = [lagged_corr(u_new, u_recon_demeaned, l) for l in lags]
    best_lag   = lags[argmax(xcorr_vals)]
    
    println("\nLag (days) of best correlation between original & reconstruction: ", best_lag)
    if best_lag == 0
        println("  -> Peak alignment at lag 0 confirms the modes are correctly phased.")
    else
        println("  -> Peak alignment is offset by $best_lag day(s) — there is a residual phase error.")
    end
    
    plot_phase_check = Plots.plot(lags, xcorr_vals,
        xlabel     = "Lag (days)",
        ylabel     = "Correlation",
        title      = "Cross-correlation: Original vs Reconstructed",
        label      = "corr(lag)",
        lw         = 2,
        seriestype = :line
    )
    Plots.vline!(plot_phase_check, [0], label="lag = 0", linestyle=:dash, color=:black)
    display(plot_phase_check)
    

    #----------------------
    #USING OTHER DATA POINTS
    #----------------------

        # Re-open (you closed it earlier) and pull u at every latitude, fixed lon/depth
    daily_data_nc = NCDataset("eastward_vel_daily_data/daily_eastward_sea_velocity_2000_to_2026.nc")
    i_lon = argmin(abs.(daily_longitude .- target_longitude))
    i_dep = argmin(abs.(daily_depth     .- target_depth))

    # u_by_lat: (latitude x time)
    u_by_lat = Float64.(coalesce.(daily_data_nc["uo"][i_lon, :, i_dep, time_indices], NaN))
    close(daily_data_nc)

    Nlat = length(daily_latitude)

    # For each latitude, FFT the time series and pull out amplitude+phase 
    # at the SAME period bin you already flagged as your Yanai candidate (~18 days)
    target_period = 18.0  # days -- swap in whatever your band search found
    target_freq   = 1.0 / target_period

    amp_by_lat   = zeros(Nlat)
    phase_by_lat = zeros(Nlat)

    for j in 1:Nlat
        ts = u_by_lat[j, :]
        valid_mean = mean(filter(!isnan, ts))
        ts_clean = map(v -> isnan(v) ? valid_mean : v, ts)
        ts_demeaned = ts_clean .- mean(ts_clean)
        ts_windowed = ts_demeaned .* hann   # reuse your existing Hann window

        ffti = fft(ts_windowed)
        # find the bin closest to target_freq
        i_target = argmin(abs.(Frequencies .- target_freq))

        amp_by_lat[j]   = (2 * abs(ffti[i_target]) / N_days) / window_correction
        phase_by_lat[j] = angle(ffti[i_target])
    end

    plot_meridional = Plots.plot(daily_latitude, amp_by_lat,
        xlabel = "Latitude",
        ylabel = "Amplitude at $(target_period)d period (m/s)",
        title  = "Meridional structure of u at ~$(target_period)-day period",
        label  = "amplitude",
        lw = 2, marker = :circle
    )
    Plots.vline!(plot_meridional, [0.0], label="Equator", linestyle=:dash, color=:black)
    display(plot_meridional)

    plot_meridional_phase = Plots.plot(daily_latitude, phase_by_lat,
        xlabel = "Latitude",
        ylabel = "Phase (rad)",
        title  = "Meridional phase structure of u at ~$(target_period)-day period",
        label  = "phase",
        lw = 2, marker = :circle
    )
    Plots.vline!(plot_meridional_phase, [0.0], label="Equator", linestyle=:dash, color=:black)
    display(plot_meridional_phase)



    #=====================================================================
    MERIDIONAL STRUCTURE SURVEY ACROSS MULTIPLE FREQUENCY BANDS
    (Yanai/inertia-gravity, TIW, intraseasonal Kelvin, annual, ENSO)
    
    Requires the daily-data variables from your main script already in
    scope: daily_longitude, daily_latitude, daily_depth, target_longitude,
    target_depth, time_indices, hann, N_days, Frequencies, window_correction
    
    NOTE: Tides (band 1, ~12-24 hr) are skipped -- your daily sampling
    has a Nyquist limit of 2 days, so diurnal/semidiurnal tides are
    unresolvable (aliased), not just noisy, in this dataset.
    =====================================================================#
    
    function meridional_structure(filename, i_lon_d, i_dep_d, time_indices,
                                hann, N_days, Frequencies, window_correction,
                                target_period)
    
        ds = NCDataset(filename)
        u_by_lat = Float64.(coalesce.(ds["uo"][i_lon_d, :, i_dep_d, time_indices], NaN))
        close(ds)
    
        Nlat = size(u_by_lat, 1)
        target_freq = 1.0 / target_period
        amp_by_lat   = zeros(Nlat)
        phase_by_lat = zeros(Nlat)
    
        for j in 1:Nlat
            ts = u_by_lat[j, :]
            valid_mean = mean(filter(!isnan, ts))
            ts_clean = map(v -> isnan(v) ? valid_mean : v, ts)
            ts_demeaned = ts_clean .- mean(ts_clean)
            ts_windowed = ts_demeaned .* hann
    
            ffti = fft(ts_windowed)
            i_target = argmin(abs.(Frequencies .- target_freq))
    
            amp_by_lat[j]   = (2 * abs(ffti[i_target]) / N_days) / window_correction
            phase_by_lat[j] = angle(ffti[i_target])
        end
    
        return amp_by_lat, phase_by_lat
    end
    
    # ---- Fixed indices (reused across all bands) ----
    daily_data_filename = "eastward_vel_daily_data/daily_eastward_sea_velocity_2000_to_2026.nc"
    i_lon_d = argmin(abs.(daily_longitude .- target_longitude))
    i_dep_d = argmin(abs.(daily_depth     .- target_depth))
    
    # ---- Band definitions: (label, representative period in days, expected structure) ----
    # Representative periods are picked from the middle/most-cited value in each band
    bands = [
        ("Yanai / inertia-gravity (3-15d)",      8.0,   "antisymmetric u (expect equatorial notch)"),
        ("TIW (17-33d)",                         18.0,  "antisymmetric u (expect equatorial notch)"),
        ("TIW broad mode (17-33d)",              33.0,  "antisymmetric u (expect equatorial notch)"),
        ("Intraseasonal Kelvin / MJO (30-90d)",  60.0,  "symmetric u, NO notch, single peak at equator"),
        ("Annual cycle (~365d)",                 365.0, "not a trapped wave -- likely broad/symmetric-ish, wind-forced"),
        ("ENSO (2-7 yr)",                        1460.0,"symmetric-ish, very low-frequency -- few cycles in record, treat cautiously"),
    ]
    
    results = Dict{String, NamedTuple}()
    
    for (label, T, expectation) in bands
        println("\n=== $label  (target period = $(T) days) ===")
        println("Theoretical expectation: $expectation")
    
        amp, phase = meridional_structure(daily_data_filename, i_lon_d, i_dep_d,
                                        time_indices, hann, N_days, Frequencies,
                                        window_correction, T)
        results[label] = (amp=amp, phase=phase, period=T)
    
        plot_amp = Plots.plot(daily_latitude, amp,
            xlabel = "Latitude",
            ylabel = "Amplitude (m/s)",
            title  = "$label\nAmplitude at ~$(T)d period",
            label  = "amplitude",
            lw = 2, marker = :circle, markersize = 3,
            size = (800, 400)
        )
        Plots.vline!(plot_amp, [0.0], label="Equator", linestyle=:dash, color=:black)
        display(plot_amp)
    
        plot_phase = Plots.plot(daily_latitude, phase,
            xlabel = "Latitude",
            ylabel = "Phase (rad)",
            title  = "$label\nPhase at ~$(T)d period",
            label  = "phase",
            lw = 2, marker = :circle, markersize = 3,
            size = (800, 400)
        )
        Plots.vline!(plot_phase, [0.0], label="Equator", linestyle=:dash, color=:black)
        display(plot_phase)
    end
