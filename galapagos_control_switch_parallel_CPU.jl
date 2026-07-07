using Pkg
using Oceananigans
using Oceananigans.Units
using CUDA: has_cuda_gpu, @allowscalar, CuArray
using Statistics: mean
using Oceanostics
using NCDatasets
using Interpolations
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.Fields: FunctionField
using Oceananigans: Callback, IterationInterval
using Oceanostics.ProgressMessengers: SingleLineMessenger
using Dates
using DataFrames
using XLSX

#-------------------------------------------------------------------------------------
# CONTROL BOARD
#-------------------------------------------------------------------------------------
struct RunParameters
    bathymetry_mode::Int   # 0 = No bathymetry, 1 = Gaussian bathymetry, 2 = Real bathymetry
    wind::Int              # 0 = no wind, 1 = wind from 4-year netCDF
    beta_switch::Int       # 0 = no beta, 1 = beta plane
    H_S_flux::Int          # 0 = no flux, 1 = heat/salt flux diagnostics
    smoothing::Int         # 0 = raw bathymetry, 1 = gaussian smoothed
    EUC_model::Int         # 0 = constant forcing, 1 = fourier-based (not yet implemented)
    model_type::String     # "hydrostatic" or "nonhydrostatic"
end

#-------------------------------------------------------------------------------------
# TOP-LEVEL PURE FUNCTIONS
# These live outside run_simulation on purpose: a closure defined inside a function,
# whose capture is only assigned on some branches, gets "boxed" by Julia (Core.Box),
# which crashes Oceananigans' internal forcing introspection with UndefRefError.
# Top-level functions can never be boxed, since they have no enclosing scope to capture from.
#-------------------------------------------------------------------------------------

@inline Sₗ(z) = 35.0 + (35.0 - 34.7) * (z / 500)

@inline function U_EUC(y, z, p)
    if p.EUC_model == 0
        return p.Umaxᵥ * exp(-(y - p.yₒᵥ)^2 / (2*p.σ_yᵥ^2)) * exp(-(z - p.zₒᵥ)^2 / (2*p.σ_zᵥ^2))
    else
        return 0.0   # Fourier-based EUC model not implemented yet
    end
end

@inline Teast(z, p) = (22 - 10) * z / 500 + 22
@inline Twest(z, p) = (22 - 10) * z / 500 + 22
@inline Seast(x, y, z, p) = Sₗ(z)
@inline Swest(x, y, z, p) = Sₗ(z)
@inline Ueast(y, z, p) = U_EUC(y, z, p)
@inline Uwest(y, z, p) = U_EUC(y, z, p)

@inline function west_mask(x, y, z, p)
    x1 = 50000.0
    return ifelse(0.0 <= x <= x1, x / x1, 0.0)
end

@inline function east_mask(x, y, z, p)
    x1 = 50000.0
    x2 = p.Lx - x1
    return ifelse(x2 <= x <= p.Lx, 1.0 - (p.Lx - x) / x1, 0.0)
end

@inline function sponge_T(i, j, k, grid, clock, model_fields, p)
    T = @inbounds model_fields.T[i, j, k]
    x = xnode(i, j, k, grid, Center(), Center(), Center())
    y = ynode(i, j, k, grid, Center(), Center(), Center())
    z = znode(i, j, k, grid, Center(), Center(), Center())
    return -east_mask(x, y, z, p) / p.σ * (T - Teast(z, p)) -
            west_mask(x, y, z, p) / p.σ * (T - Twest(z, p))
end

@inline function sponge_S(i, j, k, grid, clock, model_fields, p)
    S = @inbounds model_fields.S[i, j, k]
    x = xnode(i, j, k, grid, Center(), Center(), Center())
    y = ynode(i, j, k, grid, Center(), Center(), Center())
    z = znode(i, j, k, grid, Center(), Center(), Center())
    return -east_mask(x, y, z, p) / p.σ * (S - Seast(x, y, z, p)) -
            west_mask(x, y, z, p) / p.σ * (S - Swest(x, y, z, p))
end

@inline function sponge_u(i, j, k, grid, clock, model_fields, p)
    u = @inbounds model_fields.u[i, j, k]
    x = xnode(i, j, k, grid, Face(), Center(), Center())
    y = ynode(i, j, k, grid, Face(), Center(), Center())
    z = znode(i, j, k, grid, Face(), Center(), Center())
    return -east_mask(x, y, z, p) / p.σ * (u - Ueast(y, z, p)) -
            west_mask(x, y, z, p) / p.σ * (u - Uwest(y, z, p))
end

function make_gaussian_kernel(sigma=1.0)
    offsets = -1:1
    kernel = [exp(-(x^2 + y^2) / (2*sigma^2)) for y in offsets, x in offsets]
    return kernel ./ sum(kernel)
end

function apply_gaussian(data::AbstractMatrix, sigma=1.0)
    kernel = make_gaussian_kernel(sigma)
    nrows, ncols = size(data)
    output = copy(data)
    for j in 2:ncols-1, i in 2:nrows-1
        patch = @view data[i-1:i+1, j-1:j+1]
        valid = .!ismissing.(patch) .& .!isnan.(patch)
        w = kernel[valid]
        vals = Float64.(patch[valid])
        if !isempty(vals)
            output[i, j] = sum(w .* vals) / sum(w)
        end
    end
    return output
end

#-------------------------------------------------------------------------------------
# MAIN SIMULATION FUNCTION
#-------------------------------------------------------------------------------------
function run_simulation(p::RunParameters)
    bathymetry_mode      = p.bathymetry_mode
    wind                 = p.wind
    beta_switch          = p.beta_switch
    H_S_flux             = p.H_S_flux
    Smoothing_bathymetry = p.smoothing
    EUC_model            = p.EUC_model
    model_type           = p.model_type

    if wind == 0
        @info "no wind being used"
    elseif wind == 1
        @info "loading four years of wind data from netCDF file"
        ds = NCDataset("wind_data_4_years.nc")
        u10 = ds["u10_reg"][:]
        v10 = ds["v10_reg"][:]
        lat1 = ds["lat"][:]
        lon1 = ds["lon"][:]
        close(ds)
    end

    rundir = @__DIR__
    overwrite_existing = true
    interpolated_IC = false
    mass_flux = false
    LES = true
    ext_forcing = true
    arch = CPU()

    Lz = 500.0

    bottom = if bathymetry_mode == 0
        @info "Using no bathymetry (flat bottom)"
        Lx_real = 1000e3
        Ly_real = 500e3
        (x, y) -> -500.0

    elseif bathymetry_mode == 1
        @info "Using gaussian bathymetry"
        Lx_real = 1000e3
        Ly_real = 500e3
        (x, y) -> -500.0 + 560.0 * exp(-(x - Lx_real/2)^2 / (2*(30e3)^2)) * exp(-y^2 / (2*(30e3)^2))

    elseif bathymetry_mode == 2
        @info "Using real bathymetry"
        ds = NCDataset("galap.nc")
        lon = ds["x"][:]
        lat = ds["y"][:]
        zflat = ds["z"][:]
        close(ds)

        deg_per_meter = 1 / 111e3
        nx = length(lon)
        ny = length(lat)
        depth = reshape(zflat, nx, ny)

        if Smoothing_bathymetry == 0
            @info "Using non-smoothed real island bathymetry"
        elseif Smoothing_bathymetry == 1
            sigma_val = 1.0
            depth = apply_gaussian(depth, sigma_val)
            @info "Using gaussian filter with sigma of $sigma_val"
        end

        depth = min.(depth, 0)
        depth[(-10 .< depth) .& (depth .< 0)] .= 0

        Lx_real = (maximum(lon) - minimum(lon)) * 111e3
        Ly_real = (maximum(lat) - minimum(lat)) * 111e3

        sponge_m = 50000.0
        dlon = abs(lon[2] - lon[1])
        sponge_cols = round(Int, sponge_m * deg_per_meter / dlon)
        band_width = 5
        west_reference_depth = mean(depth[sponge_cols:sponge_cols+band_width, :])
        east_reference_depth = mean(depth[end-sponge_cols-band_width:end-sponge_cols, :])
        depth[1:sponge_cols, :] .= west_reference_depth
        depth[end-sponge_cols:end, :] .= east_reference_depth

        itp = extrapolate(
            interpolate((lat, lon), collect(depth'), Gridded(Linear())),
            Interpolations.Flat()
        )

        lon_min = minimum(lon)
        lat_min = minimum(lat)
        y_offset = 75000.0
        lon_from_x(x) = lon_min + x * deg_per_meter
        lat_from_y(y) = lat_min + (y + Ly_real/2 - y_offset) * deg_per_meter

        (x, y) -> begin
            val = itp(lat_from_y(y), lon_from_x(x))
            return isnan(val) || ismissing(val) ? -Lz : Float64(val)
        end

    else
        @warn "Unknown bathymetry_mode; defaulting to gaussian bathymetry"
        Lx_real = 1000e3
        Ly_real = 500e3
        (x, y) -> -500.0 + 560.0 * exp(-(x - Lx_real/2)^2 / (2*(30e3)^2)) * exp(-y^2 / (2*(30e3)^2))
    end

    params = (; Lx = Lx_real, Ly = Ly_real, Lz = Lz,
              Nx = 30, Ny = 30, Nz = 30,
              N²₀ = 2e-4, σ = 40000.0seconds,
              u_b = 0.0, v_b = 0.0,
              EUC_model = EUC_model,
              Umaxᵥ = 0.5, zₒᵥ = -75.0, yₒᵥ = 0.0, σ_zᵥ = 20.0, σ_yᵥ = 55600.0)

    underlying_grid = RectilinearGrid(arch,
                        size = (params.Nx, params.Ny, params.Nz),
                        x = (0, params.Lx),
                        y = (-params.Ly/2, +params.Ly/2),
                        z = (-params.Lz, 0),
                        halo = (4, 4, 4),
                        topology = (Oceananigans.Grids.Periodic, Oceananigans.Grids.Bounded, Oceananigans.Grids.Bounded))

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))
    @info "Grid" grid

    cᴰ = 2.5e-3
    ρₐ = 1.225
    ρₒ = 1028
    Qu = -ρₐ / ρₒ * cᴰ * params.u_b * abs(params.u_b)
    Qv = -ρₐ / ρₒ * cᴰ * params.v_b * abs(params.v_b)

    FT = Forcing(sponge_T, discrete_form=true, parameters=params)
    FS = Forcing(sponge_S, discrete_form=true, parameters=params)
    FU = Forcing(sponge_u, discrete_form=true, parameters=params)
    forcing = (T=FT, S=FS, u=FU)

    T_bcs = FieldBoundaryConditions()
    S_bcs = FieldBoundaryConditions()
    u_bcs = FieldBoundaryConditions()
    v_bcs = FieldBoundaryConditions()
    w_bcs = FieldBoundaryConditions()
    boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs, T=T_bcs, S=S_bcs)

    closure = AnisotropicMinimumDissipation()

    if beta_switch == 0
        @info "No beta plane"
        coriolis = BetaPlane(latitude=0)
    elseif beta_switch == 1
        @info "Using beta plane"
        β = 2.28e-11
        coriolis = BetaPlane(β=β, latitude=0)
    else
        @warn "Unknown beta_switch value; defaulting to beta plane with latitude=0"
        β = 2.28e-11
        coriolis = BetaPlane(β=β, latitude=0)
    end

    if model_type == "hydrostatic"
        @info "Using HydrostaticFreeSurfaceModel"
        model = HydrostaticFreeSurfaceModel(grid,
                    tracers = (:T, :S),
                    buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(
                        thermal_expansion = 3.87e-5, haline_contraction = 7.86e-4)),
                    momentum_advection = WENO(),
                    tracer_advection = WENO(),
                    coriolis = coriolis,
                    closure = closure,
                    forcing = forcing,
                    boundary_conditions = boundary_conditions)
    elseif model_type == "nonhydrostatic"
        @warn "Using NonhydrostaticModel"
        model = NonhydrostaticModel(grid,
                    tracers = (:T, :S),
                    buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(
                        thermal_expansion = 3.87e-5, haline_contraction = 7.86e-4)),
                    momentum_advection = WENO(),
                    tracer_advection = WENO(),
                    coriolis = coriolis,
                    closure = closure,
                    forcing = forcing,
                    boundary_conditions = boundary_conditions)
    else
        error("Unknown model_type: $model_type")
    end

    @info "Model" model

    Δt₀ = 1/2 * minimum_yspacing(grid)
    simulation = Simulation(model, Δt=Δt₀, stop_time = 365days)

    wizard = TimeStepWizard(cfl=0.5, max_change=1.02, min_change=0.5)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(2))

    callback_interval = 86400seconds
    progress = SingleLineMessenger()
    simulation.callbacks[:progress] = Callback(progress, TimeInterval(callback_interval))

    @info "Simulation" simulation

    u, v, w = model.velocities
    T = model.tracers.T
    S = model.tracers.S

    if interpolated_IC
        filename = "IC_real_bathymetry_1year.nc"
        @info "Imposing initial conditions from existing NetCDF file $filename"
        ds_ic = NCDataset(filename)
        u_data = ds_ic["u"][:, :, :, end]
        v_data = ds_ic["v"][:, :, :, end]
        w_data = ds_ic["w"][:, :, :, end]
        T_data = ds_ic["T"][:, :, :, end]
        S_data = ds_ic["S"][:, :, :, end]
        close(ds_ic)
        replace!(u_data, NaN => 0.0)
        replace!(v_data, NaN => 0.0)
        replace!(w_data, NaN => 0.0)
        replace!(T_data, NaN => 0.0)
        replace!(S_data, NaN => 0.0)
        set!(model, u=u_data, v=v_data, w=w_data, T=T_data, S=S_data)
    else
        @info "Imposing initial conditions from scratch"
        T_ic(x, y, z) = Teast(z, params)
        S_ic(x, y, z) = Seast(x, y, z, params)
        uᵢ = fill(0.0, size(u))
        vᵢ = fill(0.0, size(v))
        wᵢ = fill(0.0, size(w))
        set!(model, T=T_ic, S=S_ic, u=uᵢ, v=vᵢ, w=wᵢ)
    end

    @info "Creating output fields"

    vorticity_z = Field(∂x(v) - ∂y(u))
    KE_u = Field(@at (Center, Center, Center) 0.5 * u^2)
    KE_v = Field(@at (Center, Center, Center) 0.5 * v^2)
    KE_w = Field(@at (Center, Center, Center) 0.5 * w^2)
    KE_total = Field(@at (Center, Center, Center) 0.5 * (u^2 + v^2 + w^2))

    Δz = params.Lz / params.Nz
    k_75m  = Int(floor((params.Lz - 75)  / Δz))
    k_150m = Int(floor((params.Lz - 150) / Δz))
    k_225m = Int(floor((params.Lz - 225) / Δz))

    if H_S_flux == 0
        @info "No heat and salt fluxes being calculated"
    elseif H_S_flux == 1
        @info "Calculating heat and salt fluxes"
        T_background(x, y, z) = (22-10)*z / 500 + 22
        S_background(x, y, z) = Sₗ(z)
        T_background_field = FunctionField((Center, Center, Center), T_background, model.grid)
        S_background_field = FunctionField((Center, Center, Center), S_background, model.grid)
        T_diff = Field(T - T_background_field)
        S_diff = Field(S - S_background_field)
        wT_difference = Field(@at (Center, Center, Center) w * T_diff)
        wS_difference = Field(@at (Center, Center, Center) w * S_diff)
        ∫wT_difference_up = Integral(wT_difference)
        ∫wS_difference_up = Integral(wS_difference)
    end

    saved_output_prefix = mass_flux ? "iceplume" : "iceplume_nomf"
    checkpointer_prefix = "checkpoint_" * saved_output_prefix

    bathy_tag = bathymetry_mode == 0 ? "no_bathymetry" :
                bathymetry_mode == 1 ? "gaussian_bathymetry" : "real_bathymetry"
    windtag   = wind == 0 ? "no_wind" : "wind"
    beta_tag  = beta_switch == 0 ? "no_beta" : "beta"

    run_timestamp = Dates.format(now(), "yyyymmdd_HHMMSS_sss")
    run_tag = "$(bathy_tag)_$(beta_tag)_$(windtag)_$(run_timestamp)"
    runs_dir = joinpath(rundir, "runs")
    mkpath(runs_dir)
    output_dir = joinpath(runs_dir, run_tag)
    mkpath(output_dir)

    metadata = DataFrame(
        RunTag = run_tag, Timestamp = run_timestamp, OutputFolder = output_dir,
        ModelType = model_type, BathymetryMode = bathymetry_mode, Bathymetry = bathy_tag,
        Wind = wind, WindTag = windtag, BetaPlane = beta_switch, BetaTag = beta_tag,
        EUCModel = EUC_model, HeatSaltFlux = H_S_flux, SmoothedBathymetry = Smoothing_bathymetry,
        InterpolatedIC = interpolated_IC, ExternalForcing = ext_forcing, Architecture = string(arch),
        Nx = params.Nx, Ny = params.Ny, Nz = params.Nz,
        Lx = params.Lx, Ly = params.Ly, Lz = params.Lz,
        Stratification = params.N²₀, InitialDeltaT = string(Δt₀),
        StopTime = string(simulation.stop_time), OutputPrefix = saved_output_prefix
    )

    metadata_filename = joinpath(output_dir, "metadata.xlsx")
    XLSX.openxlsx(metadata_filename, mode="w") do xf
        sheet = xf[1]
        XLSX.rename!(sheet, "Metadata")
        for (j, name) in enumerate(names(metadata))
            sheet[1, j] = name
        end
        for (j, value) in enumerate(metadata[1, :])
            sheet[2, j] = string(value)
        end
    end
    @info "Metadata written to $(metadata_filename)"

    common_fields = (; u, v, w, T, S, vorticity_z, KE_u, KE_v, KE_w, KE_total)

    simulation.output_writers[:surface_slice_writer] = NetCDFWriter(model, common_fields;
        filename = joinpath(output_dir, "top_.nc"),
        schedule = TimeInterval(8640seconds), indices = (:, :, params.Nz),
        overwrite_existing = overwrite_existing)

    simulation.output_writers[:y_slice_writer] = NetCDFWriter(model, common_fields;
        filename = joinpath(output_dir, "midy.nc"),
        schedule = TimeInterval(8640seconds), indices = (:, Int(params.Ny/2), :),
        overwrite_existing = overwrite_existing)

    simulation.output_writers[:xy_75_depth_writer] = NetCDFWriter(model, common_fields;
        filename = joinpath(output_dir, "upwelling_75m.nc"),
        schedule = TimeInterval(8640seconds), indices = (:, :, k_75m),
        overwrite_existing = overwrite_existing)

    simulation.output_writers[:xy_150_depth_writer] = NetCDFWriter(model, common_fields;
        filename = joinpath(output_dir, "upwelling_150m.nc"),
        schedule = TimeInterval(8640seconds), indices = (:, :, k_150m),
        overwrite_existing = overwrite_existing)

    simulation.output_writers[:xy_225_depth_writer] = NetCDFWriter(model, common_fields;
        filename = joinpath(output_dir, "upwelling_225m.nc"),
        schedule = TimeInterval(8640seconds), indices = (:, :, k_225m),
        overwrite_existing = overwrite_existing)

    if H_S_flux == 1
        flux_fields = (; wT_difference, wS_difference, ∫wT_difference_up, ∫wS_difference_up)
        simulation.output_writers[:flux_writer_75] = NetCDFWriter(model, flux_fields;
            filename = joinpath(output_dir, "fluxes_75m.nc"),
            schedule = TimeInterval(8640seconds), indices = (:, :, k_75m),
            overwrite_existing = overwrite_existing)
        simulation.output_writers[:flux_writer_150] = NetCDFWriter(model, flux_fields;
            filename = joinpath(output_dir, "fluxes_150m.nc"),
            schedule = TimeInterval(8640seconds), indices = (:, :, k_150m),
            overwrite_existing = overwrite_existing)
        simulation.output_writers[:flux_writer_225] = NetCDFWriter(model, flux_fields;
            filename = joinpath(output_dir, "fluxes_225m.nc"),
            schedule = TimeInterval(8640seconds), indices = (:, :, k_225m),
            overwrite_existing = overwrite_existing)
    end

    simulation.output_writers[:IC_writer] = NetCDFWriter(model, common_fields;
        filename = joinpath(output_dir, "IC.nc"),
        schedule = TimeInterval(365days),
        overwrite_existing = overwrite_existing)
    #=
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule = TimeInterval(8640seconds),
        prefix = checkpointer_prefix,
        cleanup = true)
    =#
    run!(simulation; pickup=false)

    return output_dir
end

#-------------------------------------------------------------------------------------
function build_master_metadata(rundir)
    runs_dir = joinpath(rundir, "runs")
    isdir(runs_dir) || return DataFrame()
    folders = sort(filter(isdir, joinpath.(runs_dir, readdir(runs_dir))))
    master = DataFrame()
    for folder in folders
        file = joinpath(folder, "metadata.xlsx")
        if isfile(file)
            df = DataFrame(XLSX.readtable(file, "Metadata"))
            append!(master, df)
        end
    end
    if isempty(master)
        @warn "No metadata.xlsx files found in $runs_dir — skipping master metadata write."
        return master
    end
    masterfile = joinpath(runs_dir, "metadata_master.xlsx")
    XLSX.writetable(masterfile, collect(eachcol(master)), names(master); overwrite=true)
    println("Master metadata written to: $masterfile")
    return master
end