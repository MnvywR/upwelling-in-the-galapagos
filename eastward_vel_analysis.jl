    using Pkg
    using NCDatasets 
    using Plots
    using FFTW
    using Dates
    using Statistics

    monthly_data = NCDataset("eastward_sea_velocity.nc")
    var_names = keys(monthly_data)
    
    show(monthly_data)
    show("----------------------------------------")
    show(var_names)


    depth = monthly_data["depth"][:] 
    longitude = monthly_data["longitude"][:]
    latitude = monthly_data["latitude"][:]
    
    # Filter data for specific depth, longitude, and latitude
    target_depth = 77.85385f0  # corresponds to 75m
    target_longitude = -94.99999f0 #corresponds to 95W, which is the longitude of the Galapagos Islands. I used -94.99999 instead of -95 because the exact value of -95 is not present in the longitude array, but -94.99999 is very close and is present in the array.
    target_latitude = 0.33333334f0 # corresponds to 0.33N, which is the latitude of the Galapagos Islands. I used 0.33333334 instead of 0.33 because the exact value of 0.33 is not present in the latitude array, but 0.33333334 is very close and is present in the array.

    # Find the indices of the nearest longitude, latitude, and depth, which are index 21 for depth, index 47 for lat, index 37 for longitude.

    i_lon = argmin(abs.(longitude .- target_longitude))
    i_lat = argmin(abs.(latitude  .- target_latitude))
    i_dep = argmin(abs.(depth     .- target_depth))

    new_data = data["uo"][i_lon, i_lat, i_dep, :]    

    plot1 = Plots.plot(new_data, title="Eastward Sea Velocity at 75m depth, 95W, 0.33N", xlabel="Time (months)", ylabel="Velocity (m/s)")
    display(plot1)

    #---------------------

    daily_data = NCDataset("eastward_sea_velocity_daily.nc")

    daily_depth = daily_data["depth"][:] 
    daily_longitude = daily_data["longitude"][:]
    daily_latitude = daily_data["latitude"][:]
    
    # Find the indices of the nearest longitude, latitude, and depth, which are index 21 for depth, index 47 for lat, index 37 for longitude.
    i_lon = argmin(abs.(daily_longitude .- target_longitude))
    i_lat = argmin(abs.(daily_latitude  .- target_latitude))
    i_dep = argmin(abs.(daily_depth     .- target_depth))


    daily_time = daily_data["time"][:]

    # Filter to first year (2018) — adjust as needed
    start_date   = DateTime(2018, 1, 1)
    end_date     = DateTime(2018, 12, 31)
    time_mask    = (daily_time .>= start_date) .& (daily_time .<= end_date)
    time_indices = findall(time_mask)


    # u dims: (longitude × latitude × depth × time) 
    new_daily_data = daily_data["uo"][i_lon, i_lat, i_dep, time_indices]

    # Clean missing
    u_clean = Float64.(coalesce.(new_daily_data, mean(skipmissing(new_daily_data))))

    plot_daily1 = Plots.plot(
        daily_time[time_indices], u_clean,
        title  = "Eastward Sea Velocity at 78m, 95°W, 0.33°N (2018)",
        xlabel = "Date",
        ylabel = "Velocity (m/s)",
        label  = "u",
        lw     = 1.5
    )
    plot_daily1 = Plots.plot(new_daily_data, title="Eastward Sea Velocity at 75m depth, 95W, 0.33N (2 years)", xlabel="Time (days)", ylabel="Velocity (m/s)")
    display(plot_daily1)


    #---------------------
    #SEE FIRST DOMINANT MODE IN THE FFT
    #---------------------
    fs = 1.0  # daily sampling
    u_clean = Float64.(coalesce.(new_daily_data, mean(skipmissing(new_daily_data)))) # Clean missing values by replacing with mean

    u_detrended = u_clean .- mean(u_clean)  # Detrend by removing mean
    N = length(u_detrended)

    #FFT
    U_modes = fft(u_detrended)
    freqs = (0:N-1) * (fs / N)  # Frequencys in cycles per day

    #plot FFT magnitude
    plot_fft = Plots.plot(freqs[1:div(N, 2)], abs.(U_modes[1:div(N, 2)]), 
        title="FFT of Eastward Sea Velocity (2018)", 
        xlabel="Frequency (cycles/day)", 
        ylabel="Magnitude", 
        label="FFT", 
        lw=1.5)
    display(plot_fft)

    i_dominant = argmax(abs.(U_modes[1:div(N, 2)]))

    f_dominant = freqs[i_dominant]

    A_dominant = 2 * abs(U_modes[i_dominant]) / N

    φ_dominant = angle(U_modes[i_dominant])

    t = collect(0:N-1)

    u_reconstructed = mean(u_clean) .+ A_dominant .* cos.(2π .* f_dominant .* t .+ φ_dominant)


    plot_reconstructed = Plots.plot(
        daily_time[time_indices], u_clean,
        title  = "EUC Velocity + Dominant Mode Fit (2018)",
        xlabel = "Date",
        ylabel = "Velocity (m/s)",
        label  = "Original data",
        lw     = 1.0,
        alpha  = 0.5,
        color  = :steelblue
    )

    Plots.plot!(plot_reconstructed,        # ← the ! means "add to existing plot"
    daily_time[time_indices], u_reconstructed,
    label  = "Dominant mode  (T=$(round(1/f_dominant, digits=0))d, A=$(round(A_dominant, digits=3)) m/s)",
    lw     = 2.0,
    color  = :red,
    ls     = :dash
    )
    display(plot_reconstructed)

    #---------------------
    #MULTIPLE DOMINANT MODES
    #---------------------
    # Identify top 3 modes
    top_n = 3

    # only search first half (positive frequencies, skip k=0 DC at index 1)
    power_onesided = abs.(U_modes[2:div(N,2)]).^2   # skip index 1 (DC)
    top_idx = sortperm(power_onesided, rev=true)[1:top_n]  # indices into power_onesided
    top_idx_fft = top_idx .+ 1   # shift back to U_modes indexing (since we skipped index 1)

    println("Top $top_n modes:")
    for i in top_idx_fft
        A_i = (2.0/N) * abs(U_modes[i])
        f_i = freqs[i]
        T_i = 1.0 / f_i
        φ_i = angle(U_modes[i])
        println("  period=$(round(T_i, digits=1))d  amplitude=$(round(A_i, digits=4)) m/s")
    end

    #recontruct signal using top modes
    t = collect(0:N-1)
    
    u_reconstructed_multi = fill(mean(u_clean), N)   # start with mean

    for i in top_idx_fft
        A_i = (2.0/N) * abs(U_modes[i])
        f_i = freqs[i]
        φ_i = angle(U_modes[i])
        u_reconstructed_multi .+= A_i .* cos.(2π .* f_i .* t .+ φ_i)
    end

    
    plot_reconstructed = Plots.plot(
    daily_time[time_indices], u_clean,
    title  = "EUC Velocity + top-$top_n mode reconstruction (2018)",
    xlabel = "Date",
    ylabel = "Velocity (m/s)",
    label  = "Original data",
    lw     = 1.0,
    alpha  = 0.5,
    color  = :steelblue
    )
    Plots.plot!(plot_reconstructed,
        daily_time[time_indices], u_reconstructed_multi[:],
        label  = "Top $top_n modes",
        lw     = 2.0,
        color  = :red
    )
    display(plot_reconstructed)