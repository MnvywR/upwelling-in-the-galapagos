# =============================================================================
#  WIND DATA ANALYSIS — GALAPAGOS ERA5
#  Five approaches to characterizing annual wind statistics
# =============================================================================

using NCDatasets
using Statistics
using CairoMakie
using Dates
using Printf

# ---------------------------------------------------------------------------
#  0. CONFIG
# ---------------------------------------------------------------------------

raw_path = "wind_data_4_years.nc"

# ---------------------------------------------------------------------------
#  1. LOAD AND RESHAPE
#  ERA5 sub-region GRIB → CDO → NetCDF stores data as [npts, ntime]
#  where lat[k] / lon[k] are the coordinates of each flattened grid point.
#  The underlying grid IS regular, so we recover (nlon × nlat) from
#  unique(lat) × unique(lon) and reshape — no interpolation needed.
# ---------------------------------------------------------------------------

println("Loading ERA5 data...")
ds = NCDataset(raw_path)

println("  Variables in file:")
for (k, v) in ds
    println("    $k  size=$(size(v))  dims=$(dimnames(v))")
end

function read_var(ds, name)
    return Array(ds[name])
end

u10_raw  = read_var(ds, "10u")   # [nlon, nlat, ntime]
v10_raw  = read_var(ds, "10v")   # [nlon, nlat, ntime]
lat_raw  = read_var(ds, "lat")   # [nlat]
lon_raw  = read_var(ds, "lon")   # [nlon]
close(ds)

nlon  = length(lon_raw)
nlat  = length(lat_raw)
ntime = size(u10_raw, 3)

println("  nlon=$nlon  nlat=$nlat  ntime=$ntime")

# ---- Coordinate arrays ----
lon_reg = Float64.(vec(lon_raw))
lat_reg = Float64.(vec(lat_raw))

if maximum(lon_reg) > 180
    lon_reg .-= 360
    @info "Converted longitudes [0,360] → [-180,180]"
end

println("  Lon range: $(minimum(lon_reg))° → $(maximum(lon_reg))°")
println("  Lat range: $(minimum(lat_reg))° → $(maximum(lat_reg))°")

# ---- Data is already [nlon, nlat, ntime] — no reshape needed ----
U = Float64.(u10_raw)
V = Float64.(v10_raw)

# Wind speed and direction
spd = sqrt.(U.^2 .+ V.^2)
dir = atand.(U, V)   # signed: 0=N, 90=E, ±180=S, -90=W

println("Done loading. Starting analyses...\n")

# ===========================================================================
#  APPROACH 1:  TIME-MEAN FIELD
#  One (u,v) vector per grid box averaged over the full record.
# ===========================================================================

println("=== APPROACH 1: Time-mean spatial field ===")

U_mean   = mapslices(x -> mean(filter(!isnan, x)), U, dims=3)[:, :, 1]
V_mean   = mapslices(x -> mean(filter(!isnan, x)), V, dims=3)[:, :, 1]
spd_mean = sqrt.(U_mean.^2 .+ V_mean.^2)
dir_mean = atand.(U_mean, V_mean)

fig1 = Figure(resolution=(1200, 500))

ax1 = Axis(fig1[1,1], xlabel="Longitude (°)", ylabel="Latitude (°)",
           title="Annual mean wind speed (m/s)")
hm1 = heatmap!(ax1, lon_reg, lat_reg, spd_mean, colormap=:viridis)
Colorbar(fig1[1,2], hm1, label="Speed (m/s)")

# Overlay wind vectors (subsample ~15 arrows across)
stride = max(1, div(nlon, 15))
qi = 1:stride:nlon
qj = 1:stride:nlat
u_arr  = vec(U_mean[qi, qj])
v_arr  = vec(V_mean[qi, qj])
sp_arr = sqrt.(u_arr.^2 .+ v_arr.^2)
sp_safe = copy(sp_arr)
sp_safe[isnan.(sp_safe) .| (sp_safe .== 0)] .= 1.0
arrows!(ax1,
    [lon_reg[i] for i in qi for j in qj],
    [lat_reg[j] for i in qi for j in qj],
    u_arr ./ sp_safe .* 0.1,
    v_arr ./ sp_safe .* 0.1,
    color=:white, arrowsize=8, linewidth=1)

ax2 = Axis(fig1[1,3], xlabel="Longitude (°)", ylabel="Latitude (°)",
           title="Annual mean wind direction (°)")
hm2 = heatmap!(ax2, lon_reg, lat_reg, dir_mean,
               colormap=:cyclic_mygbm_30_95_c78_n256_s25,
               colorrange=(-180, 180))
Colorbar(fig1[1,4], hm2, label="Direction (°)\n0=N, 90=E, ±180=S, -90=W")

display(fig1)

# ===========================================================================
#  APPROACH 2:  DOMAIN-AVERAGE TIME SERIES
# ===========================================================================

println("=== APPROACH 2: Domain-average time series ===")

spd_domain = [mean(filter(!isnan, vec(spd[:, :, t]))) for t in 1:ntime]
u_domain   = [mean(filter(!isnan, vec(U[:, :, t])))   for t in 1:ntime]
v_domain   = [mean(filter(!isnan, vec(V[:, :, t])))   for t in 1:ntime]

# Construct time axis from known data properties
t_axis = DateTime(2020,1,1,0) .+ Hour.(0:ntime-1)

fig2 = Figure(resolution=(1100, 500))
ax_ts = Axis(fig2[1,1], xlabel="Date",
             ylabel="Wind (m/s)", title="Domain-averaged wind time series")
lines!(ax_ts, t_axis, spd_domain, color=:steelblue, label="Speed |V|")
lines!(ax_ts, t_axis, u_domain,   color=:coral,     label="u (East)")
lines!(ax_ts, t_axis, v_domain,   color=:seagreen,  label="v (North)")
hlines!(ax_ts, [0], color=:black, linewidth=0.5, linestyle=:dash)
axislegend(ax_ts, position=:rt)

display(fig2)

# ===========================================================================
#  APPROACH 3:  DIRECTIONAL ANALYSIS (8 compass sectors)
# ===========================================================================

println("=== APPROACH 3: Directional frequency + speed ===")

dirs       = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
dir_domain = atand.(u_domain, v_domain)   # [-180, 180]

freq         = zeros(8)
speed_bydir  = [Float64[] for _ in 1:8]

for t in 1:ntime
    bin_idx = mod(round(Int, dir_domain[t] / 45), 8) + 1
    freq[bin_idx] += 1
    push!(speed_bydir[bin_idx], spd_domain[t])
end

freq_pct     = freq ./ ntime .* 100
mean_spd_dir = [isempty(s) ? 0.0 : mean(s) for s in speed_bydir]

fig3 = Figure(resolution=(900, 450))
ax3a = Axis(fig3[1,1], xlabel="Wind direction", ylabel="Frequency (%)",
            title="Wind direction frequency", xticks=(1:8, dirs))
barplot!(ax3a, 1:8, freq_pct, color=:steelblue, strokecolor=:white, strokewidth=1)

ax3b = Axis(fig3[1,2], xlabel="Wind direction", ylabel="Mean speed (m/s)",
            title="Mean wind speed by direction", xticks=(1:8, dirs))
barplot!(ax3b, 1:8, mean_spd_dir, color=:coral, strokecolor=:white, strokewidth=1)

display(fig3)

println("\n  Compass | Frequency | Mean speed")
println("  --------|-----------|----------")
for k in 1:8
    @printf "  %-7s | %8.1f%% | %8.2f m/s\n" dirs[k] freq_pct[k] mean_spd_dir[k]
end

# ===========================================================================
#  APPROACH 4:  1-DEGREE SPATIAL BINNING
# ===========================================================================

println("\n=== APPROACH 4: 1-degree spatial binning ===")

# Used only for the 1° binning labels
lon_min, lon_max = -94.0, -85.0
lat_min, lat_max =  -3.0,   3.0

lon_1deg = collect(ceil(lon_min):1.0:floor(lon_max))
lat_1deg = collect(ceil(lat_min):1.0:floor(lat_max))
n1lon, n1lat = length(lon_1deg), length(lat_1deg)

u_bins   = [Float64[] for _ in 1:n1lon, _ in 1:n1lat]
v_bins   = [Float64[] for _ in 1:n1lon, _ in 1:n1lat]
spd_bins = [Float64[] for _ in 1:n1lon, _ in 1:n1lat]

for t in 1:ntime
    for i in 1:nlon, j in 1:nlat
        (isnan(U[i,j,t]) || isnan(V[i,j,t])) && continue
        ii = searchsortedfirst(lon_1deg, lon_reg[i]) - 1
        jj = searchsortedfirst(lat_1deg, lat_reg[j]) - 1
        (1 <= ii <= n1lon && 1 <= jj <= n1lat) || continue
        push!(u_bins[ii, jj],   U[i,j,t])
        push!(v_bins[ii, jj],   V[i,j,t])
        push!(spd_bins[ii, jj], spd[i,j,t])
    end
end

spd_1deg_mean = [isempty(spd_bins[i,j]) ? NaN : mean(spd_bins[i,j]) for i in 1:n1lon, j in 1:n1lat]
spd_1deg_std  = [isempty(spd_bins[i,j]) ? NaN : std(spd_bins[i,j])  for i in 1:n1lon, j in 1:n1lat]
u_1deg_mean   = [isempty(u_bins[i,j])   ? NaN : mean(u_bins[i,j])   for i in 1:n1lon, j in 1:n1lat]
v_1deg_mean   = [isempty(v_bins[i,j])   ? NaN : mean(v_bins[i,j])   for i in 1:n1lon, j in 1:n1lat]

fig4 = Figure(resolution=(1100, 500))
ax4a = Axis(fig4[1,1], xlabel="Longitude (°)", ylabel="Latitude (°)",
            title="1° binned: annual mean wind speed (m/s)")
hm4a = heatmap!(ax4a, lon_1deg, lat_1deg, spd_1deg_mean, colormap=:viridis)
Colorbar(fig4[1,2], hm4a, label="Speed (m/s)")

for i in 1:n1lon, j in 1:n1lat
    isnan(spd_1deg_mean[i,j]) && continue
    text!(ax4a, lon_1deg[i], lat_1deg[j],
          text=@sprintf("%.1f", spd_1deg_mean[i,j]),
          align=(:center, :center), fontsize=10, color=:white)
end

ax4b = Axis(fig4[1,3], xlabel="Longitude (°)", ylabel="Latitude (°)",
            title="1° binned: wind speed std dev (m/s)")
hm4b = heatmap!(ax4b, lon_1deg, lat_1deg, spd_1deg_std, colormap=:plasma)
Colorbar(fig4[1,4], hm4b, label="Std dev (m/s)")

display(fig4)

println("\n  Lon | Lat | Mean spd (m/s) | Std (m/s) | Mean u | Mean v")
println("  ----|-----|----------------|-----------|--------|-------")
for i in 1:n1lon, j in 1:n1lat
    isnan(spd_1deg_mean[i,j]) && continue
    @printf "  %4.0f | %3.0f | %14.2f | %9.2f | %6.2f | %5.2f\n" lon_1deg[i] lat_1deg[j] spd_1deg_mean[i,j] spd_1deg_std[i,j] u_1deg_mean[i,j] v_1deg_mean[i,j]
end

# ===========================================================================
#  APPROACH 5:  FULL-RECORD WIND DIRECTION HISTOGRAM  (1° bins, −180→180°)
# ===========================================================================

println("\n=== APPROACH 5: Wind direction histogram (1° bins) ===")

# Flatten all grid points × all time steps into one direction vector
# dir was computed as atand.(U, V) → already in [-180, 180]
dir_flat = filter(!isnan, vec(dir))

# Build 1° bins: edges at −180.5, −179.5, …, 180.5 (360 bins)
bin_centers = collect(-180.0:1.0:180.0)   # 361 values → 360 bins between them
nbins = length(bin_centers)

counts = zeros(Int, nbins)
for d in dir_flat
    idx = clamp(round(Int, d) + 181, 1, nbins)
    counts[idx] += 1
end

freq_dir = counts ./ sum(counts) .* 100

fig5 = Figure(resolution=(1100, 450))
ax5 = Axis(fig5[1,1],
    xlabel = "Wind direction (°)  [−180=S via W, 0=N, ±180=S via E]",
    ylabel = "Frequency (%)",
    title  = "Wind direction distribution — all grid points, all years (1° bins)")

barplot!(ax5, bin_centers, freq_dir,
         color = :steelblue, strokecolor = :transparent, gap = 0.0)

for (deg, label) in [(-180,"S"), (-90,"W"), (0,"N"), (90,"E"), (180,"S")]
    vlines!(ax5, [deg], color=:black, linewidth=0.8, linestyle=:dash)
    text!(ax5, deg, maximum(freq_dir)*0.95,
          text=label, align=(:center, :top), fontsize=11, color=:black)
end

xlims!(ax5, -180, 180)
ax5.xticks = (collect(-180:30:180), string.(collect(-180:30:180)))

display(fig5)

# Print top-5 most frequent directions
top5 = sortperm(counts, rev=true)[1:5]
println("\n  Top 5 most frequent directions:")
println("  Degree | Count   | Frequency")
println("  -------|---------|----------")
for idx in top5
    @printf "  %+5d° | %7d | %8.3f%%\n" bin_centers[idx] counts[idx] freq_dir[idx]
end
