    using Revise
    using Pkg
    using Oceananigans
    using Oceananigans.Units
    using CUDA: has_cuda_gpu, @allowscalar, CuArray
    using Statistics: mean
    using Oceanostics
    using Rasters
    using NCDatasets 
    using Interpolations
    using Polynomials
    using Oceanostics
    using Oceanostics: SingleLineProgressMessenger
    using Oceananigans.Grids: xnode, ynode, znode
    using Oceananigans.Fields: FunctionField
    using Oceanostics.KineticEnergyEquation: KineticEnergy, KineticEnergyStress, DissipationRate
    using Oceanostics.FlowDiagnostics: QVelocityGradientTensorInvariant, RichardsonNumber
    using Oceanostics.TurbulentKineticEnergyEquation: ShearProductionRate, XShearProductionRate, YShearProductionRate, ZShearProductionRate
    using Statistics
    using Printf
    using Oceananigans: Callback, IterationInterval
    using Oceanostics.ProgressMessengers: SingleLineMessenger 
    using CairoMakie
    using XLSX, DataFrames
    using Dates

    #-------------------------------------------------------------------------------------
    #CONTROL BOARD
    
    # Bathymetry switch
    bathymetry_mode = 2   # 0 = No bathymetry, 1 = Gaussian bathymetry, 2 = Real bathymetry
    
    #wind switch
    wind = 0 #0 for no wind, 1 for wind (using 4 years of data from netCDF file)

    #non-beta vs beta switch
    beta_switch = 1 #0 for no beta, 1 for beta plane

    #Heat and salt flux switch
    H_S_flux = 0 #0 for no flux, 1 for flux (using linear functions of z for temperature and salinity)

    #Smoothing bathymetry switch
    Smoothing_bathymetry = 1 #0 for original bathymetry, 1 for gaussian smoothed bathymetry

    EUC_model = "constant" #Look at the function, but constant = constant forcing, fourier-based is the data based waveforms
    #-------------------------------------------------------------------------------------


    if wind==0

        @info "no wind being used"

    elseif wind==1
        
        @info "loading four years of wind data from netCDF file"
        
        ds = NCDataset("wind_data_4_years.nc") #load wind data 
        
        u10 = ds["u10_reg"][:]
        v10 = ds["v10_reg"][:]
        lat1 = ds["lat"][:]
        lon1 = ds["lon"][:]

        close(ds)

    end   

    #+++ Preamble
    rundir = @__DIR__ # `rundir` will be the directory of this file
    #---
    overwrite_existing = true  # default value
    #+++ High level options
    interpolated_IC = false
    mass_flux = false
    LES = true
    ext_forcing = true

    if has_cuda_gpu()
        arch = GPU()
    else
        arch = CPU()
    end
    #---

    function make_gaussian_kernel(sigma=1.0)
        offsets = -1:1
        kernel = [exp(-(x^2 + y^2) / (2*sigma^2)) for y in offsets, x in offsets]
        return kernel ./ sum(kernel)
    end

    function apply_gaussian(data::AbstractMatrix, sigma=1.0)
        kernel = make_gaussian_kernel(sigma)
        nrows, ncols = size(data)
        output = copy(data)   # copy so borders keep original values

        for j in 2:ncols-1
            for i in 2:nrows-1
                patch = @view data[i-1:i+1, j-1:j+1]
                valid = .!ismissing.(patch) .& .!isnan.(patch)
                w     = kernel[valid]
                vals  = Float64.(patch[valid])
                if !isempty(vals)
                    output[i, j] = sum(w .* vals) / sum(w)
                end
            end
        end

        return output
    end

    if  bathymetry_mode == 0 
        @info "Using no bathymetry (flat bottom)"
        Lx_real = 1000e3
        Ly_real = 500e3
        
        #Flat bottom
        bottom(x,y) = -500

    elseif bathymetry_mode == 1

        @info "Using gaussian bathymetry"

        #Gaussian Bathymetry of Galapagos
        #height of 500 m, 250km mean, 3e4 (30 km STD)
        Lx_real = 1000e3
        Ly_real = 500e3
        bottom(x,y) = -500 + 560 * exp( -(x-params.Lx/2)^2/(2*(30e3)^2) )* exp(-(y-0)^2/(2*(30e3)^2))

    elseif bathymetry_mode == 2
        @info "Using real bathymetry"

        #Load Bathymetry data of the Galapagos
        ds = NCDataset("galap.nc")

        lon = ds["x"][:]   # longitudesThreads.nthreads()
        lat = ds["y"][:]   # latitudes
        zflat = ds["z"][:] # depth values (flattened)
        close(ds)

        deg_per_meter = 1 / 111e3

        nx = length(lon)
        ny = length(lat)

        #Depth is [lon,lat]
        depth = reshape(zflat, nx, ny) # reshape to 2D grid (lat × lon)

        if Smoothing_bathymetry == 0
            @info "Using non-smoothed real island bathymetry"

        elseif Smoothing_bathymetry == 1
            sigma_val = 1.0
            depth = apply_gaussian(depth, sigma_val)
            @info "Using gaussian filter with sigma of $sigma_val"

        end
        

        depth = min.(depth, 0) #clipping depth for land above zero (i.e. sea level)
        depth[(-10 .< depth) .& (depth .< 0)] .= 0 #clipping between 20-0 depth for gradiant spikes

        Lx_real = (maximum(lon) - minimum(lon)) * 111e3
        Ly_real = (maximum(lat) - minimum(lat)) * 111e3


        sponge_m = 50000.0
        dlon = abs(lon[2] - lon[1]) #Grid points equating to 50km
        sponge_cols = round(Int, sponge_m * deg_per_meter / dlon)

        #West sponge; computing mean depth
        band_width = 5
        west_reference_depth = mean(depth[sponge_cols: sponge_cols+band_width, :])
        east_reference_depth = mean(depth[end-sponge_cols-band_width : end-sponge_cols, :])

        depth[1:sponge_cols, :] .= west_reference_depth
        depth[end-sponge_cols:end, :] .= east_reference_depth

        #Interpolator is taking latitude and longitude (i.e., y and x)

        itp = extrapolate(
            interpolate((lat, lon), collect(depth'), Gridded(Linear())),
            Interpolations.Flat()
        )

        #island_lat = 0
        lon_from_x(x) = minimum(lon) + x * deg_per_meter

        y_offset = 75000  # meters — tune this until island is centered

        lat_from_y(y) = minimum(lat) + (y + Ly_real/2 - y_offset) * deg_per_meter

        #bottom(x, y) = itp(lat_from_y(y), lon_from_x(x))
        function bottom(x, y)
            val = itp(lat_from_y(y), lon_from_x(x))
            return isnan(val) || ismissing(val) ? -params.Lz : Float64(val)
        end
    else
        @warn "Unknown bathymetry_mode; defaulting to gaussian bathymetry"

        #Gaussian Bathymetry of Galapagos
        #height of 500 m, 250km mean, 3e4 (30 km STD)
        Lx_real = 1000e3
        Ly_real = 500e3
        bottom(x,y) = -500 + 560 * exp( -(x-params.Lx/2)^2/(2*(30e3)^2) )* exp(-(y-0)^2/(2*(30e3)^2))

    end
   

    #++++ Construct grid
    
    params = (; Lx = Lx_real,
            Ly = Ly_real,
            Lz = 500,
            Nx = 30,
            Ny = 30,
            Nz = 30,
            N²₀ = 2e-4, #  9.83/1028*2/100  1/s (stratification frequency)
            σ = 40000.0seconds, # s (relaxation timescale for sponge layer) how long we expect it to; now 1 day CHANGedto 40000 seconds (half a day)
            #uₑᵥₐᵣ = 0.00, # m/s (velocity variation along the z direction of the east boundary)
            u_b = 0.0,    # m s⁻¹, average wind velocity 10 meters above the ocean
            v_b = 0.0,    #-10    # m s⁻¹, average wind velocity 10 meters above the ocean
            )

    #changing grid i.e. Nx Ny Nz to 30 30 30 cuz laptop
    if arch == CPU() 
        params = (; params..., Nx = 30, Ny = 30, Nz = 30)
    end

    underlying_grid = RectilinearGrid(arch,
                        size = (params.Nx, params.Ny, params.Nz),
                        x = (0, params.Lx),
                        y = (-params.Ly/2, +params.Ly/2),
                        z = (-params.Lz, 0), 
                        halo = (4, 4, 4),
                        topology = (Oceananigans.Grids.Periodic, Oceananigans.Grids.Bounded, Oceananigans.Grids.Bounded))

    #----

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))

    @info "Grid" grid

    #----

    if EUC_model == "constant"
        #modeling eastward EUC velocity with the data from paper
        const Umaxᵥ = 0.5 #m/s
        const zₒᵥ = -75 #m 
        const yₒᵥ = 0 #m
        const σ_zᵥ = 20 #m #change to around 20 ish
        const σ_yᵥ = 55600 #0.5 * pi/180 * (6.371*10^6) #m = 55,600 meters #divide by two 

        @inline U₁(y) = exp(-(y-yₒᵥ)^2/(2*σ_yᵥ^2))
        @inline U₂(z)= exp(-(z-zₒᵥ)^2/(2*σ_zᵥ^2))
        @inline U_EUC(y,z) = Umaxᵥ * U₁(y) * U₂(z)
    elseif EUC_model == "fourier based"
        nothing
    end

    #Modeling temperature profile with the data from paper

    #linear salinity function for the right
    @inline Sₗ(z) = 35.0 + (35.0 - 34.7) * (z / 500)


    #++++ Conditions opposite to the ice wall (@ infinity)
    if LES
    #  b∞(z, parameters) = params.N²₀ * z # Linear stratification in the interior (far from ice face)
    #make it into functions of y
        @inline Teast(z,parameters) = (22-10)*z / 500 + 22 #22 from top to 10 from bottom over 500 meters
        @inline Seast(x,y,z,parameters) = Sₗ(z) #should I add +35 for the background??
        @inline Twest(z,parameters) =  (22-10)*z / 500 + 22 #same deal as Teast
        @inline Swest(x,y,z,parameters) = Sₗ(z)
        @inline Uwest(y,z,parameters) = U_EUC(y,z)  #m/s
        @inline Ueast(y,z,parameters) = U_EUC(y,z) #m/s
        u∞(z, parameters) = params.u_b
        v∞(z, parameters) = params.v_b
    end
    #----

    #++++ EAST BCs
  
    #++++ WEST BCs

    #----


    #surface wind stresses

    cᴰ = 2.5e-3 # dimensionless drag coefficient
    ρₐ = 1.225  # kg m⁻³, average density of air at sea-level
    ρₒ = 1028   # kg m⁻³, average density of seawater
    Qu = - ρₐ / ρₒ * cᴰ * params.u_b * abs(params.u_b) # m² s⁻²
    Qv = - ρₐ / ρₒ * cᴰ * params.v_b * abs(params.v_b) # m² s⁻²

    #++++ Drag BC for v and w
    if LES
    # const κ = 0.4 # von Karman constant
    # z₁ = first(znodes(grid, Center())) # Closest grid center to the bottom
    # cᴰ = 2.5e-3 # (κ / log(z₁ / z₀))^2 # Drag coefficient
    # x₁ₘₒ = @allowscalar xnodes(grid, Center())[1] # Closest grid center to the bottom
    # cᴰ = (κ / log(x₁ₘₒ/params.ℓ₀))^2 # Drag coefficient

        @inline drag_u(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * u
        @inline drag_v(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * v

        drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ=cᴰ,))
        drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ=cᴰ,))
    end
    #----

    #++++ West sponge layer 

    # (smoothes out the mass flux and gets rid of some of the build up of buoyancy)
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
        T   = @inbounds model_fields.T[i, j, k]
        x = xnode(i, j, k, grid, Center(), Center(), Center())
        y = ynode(i, j, k, grid, Center(), Center(), Center())
        z = znode(i, j, k, grid, Center(), Center(), Center())
        return -east_mask(x, y, z, p) / p.σ * (T - Teast(z, p)) -
                west_mask(x, y, z, p) / p.σ * (T - Twest(z, p))
    end

    @inline function sponge_S(i, j, k, grid, clock, model_fields, p)
        S   = @inbounds model_fields.S[i, j, k]
        x = xnode(i, j, k, grid, Center(), Center(), Center())
        y = ynode(i, j, k, grid, Center(), Center(), Center())
        z = znode(i, j, k, grid, Center(), Center(), Center())
        return -east_mask(x, y, z, p) / p.σ * (S - Seast(x, y, z, p)) -
                west_mask(x, y, z, p) / p.σ * (S - Swest(x, y, z, p))
    end

    @inline function sponge_u(i, j, k, grid, clock, model_fields, p)
        u   = @inbounds model_fields.u[i, j, k]
        x = xnode(i, j, k, grid, Face(), Center(), Center())
        y = ynode(i, j, k, grid, Face(), Center(), Center())
        z = znode(i, j, k, grid, Face(), Center(), Center())
        return -east_mask(x, y, z, p) / p.σ * (u - Ueast(y, z, p)) -
                west_mask(x, y, z, p) / p.σ * (u - Uwest(y, z, p))
    end

    FT = Forcing(sponge_T, discrete_form=true, parameters=params)
    FS = Forcing(sponge_S, discrete_form=true, parameters=params)
    FU = Forcing(sponge_u, discrete_form=true, parameters=params)

    forcing = (T=FT, S=FS, u=FU)

     #----


    #Flux boundary conditions -> Value '''' and turned on west and east for Tbcs/Sbcs
    T_bcs = FieldBoundaryConditions(#top = ValueBoundaryCondition(get_T0, field_dependencies=(:T, :S)), 
                                    #west = ValueBoundaryCondition(20), #want to verify if it is Twest, etc. i.e. 20, 35
                                    #east = ValueBoundaryCondition(20), # Hidden behind sponge layer
                                    )

                                    
    S_bcs = FieldBoundaryConditions(#top = ValueBoundaryCondition(get_S0, field_dependencies=(:T, :S)),
                                    #west = ValueBoundaryCondition(35),  
                                    #east = ValueBoundaryCondition(35), # Hidden behind sponge layer
                                    )
    
    u_bcs = FieldBoundaryConditions(
        #top    = FluxBoundaryCondition(Qu),   # wind stress
        #immersed = drag_bc_u,                       # bottom drag
    )
    v_bcs = FieldBoundaryConditions(
        #top    = FluxBoundaryCondition(Qv),   # wind stress
        #immersed = drag_bc_v,                       # bottom drag
    )

    #u_bcs = FieldBoundaryConditions(#west = ValueBoundaryCondition(0.3),  
                                    #east = ValueBoundaryCondition(0.3)
                                    #top = FluxBoundaryCondition(Qu),  # wind stress
                                    #bottom = drag_bc_u, # # bottom = drag_bc_u, #must change to quadratic function
                                   # )
    #v_bcs = FieldBoundaryConditions(#top = FluxBoundaryCondition(Qv),  # wind stress
                                    #bottom = drag_bc_v,
                                # east = ValueBoundaryCondition(0),
                                # west = ValueBoundaryCondition(0),
                                 #   )
    w_bcs = FieldBoundaryConditions(#east = ValueBoundaryCondition(0),
                                    #west = ValueBoundaryCondition(0),
                                    )
    boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs, T=T_bcs, S=S_bcs,)
    #----

    #++++ Construct model
    if LES
        closure = AnisotropicMinimumDissipation()
    else
        closure = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(),ν=1.8e-6, κ=(T=1.3e-7, S=7.2e-10))
    end

    #θ = 105 # degrees relative to pos. x-axis
    if beta_switch == 0
        @info "No beta plane"
        coriolis = BetaPlane(latitude=0)
    elseif beta_switch == 1
        @info "Using beta plane"
        β = 2.28e-11 # m⁻¹ s⁻¹, typical mid-latitude value for beta
        coriolis = BetaPlane(β=β,latitude=0)
    else
        @warn "Unknown beta_switch value; defaulting to beta plane with latitude=0"
        β = 2.28e-11 # m⁻¹ s⁻¹, typical mid-latitude value for beta
        coriolis = BetaPlane(β=β,latitude=0)
    end

    model = HydrostaticFreeSurfaceModel(grid, 
                                tracers = (:T, :S),
                                buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = 3.87e-5,
                                haline_contraction = 7.86e-4)), 
                                momentum_advection = WENO(),
                                tracer_advection = WENO(),
                                coriolis = coriolis,
                                closure = closure,
                                forcing = forcing,
                                boundary_conditions = boundary_conditions,
                                )

    @info "Model" model



    #----

    #++++ Create simulation

    Δt₀ = 1/2 * minimum_yspacing(grid) / 1 # / (u₁_west + 1)
    simulation = Simulation(model, Δt=Δt₀,
                            stop_time = 100days, # when to stop the simulation
    )

    
    #++++ Adapt time step
    wizard = TimeStepWizard(cfl=0.5, # How to adjust the time step
                            max_change=1.02, 
                            min_change=0.5, 
                            max_Δt=0.5/√params.N²₀) #max_Δt=0.5/√params.N²₀)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(2)) # When to adjust the time step
    #----   

    #++++ Printing to screen

    start_time = time_ns() * 1e-9
    callback_interval = 86400seconds
    progress = SingleLineMessenger()
    simulation.callbacks[:progress] = Callback(progress, TimeInterval(callback_interval))

    #----

    @info "Simulation" simulation
    #----

    #++++ Impose initial conditions
    u, v, w =  model.velocities

    T = model.tracers.T
    S = model.tracers.S

    if interpolated_IC

        filename = "IC_real_bathymetry_1year.nc"
        @info "Imposing initial conditions from existing NetCDF file $filename"
        
        ds_ic = NCDataset(filename)

        # Read the last time snapshot of each variable
        u_data = ds_ic["u"][:, :, :, end]
        v_data = ds_ic["v"][:, :, :, end]
        w_data = ds_ic["w"][:, :, :, end]
        T_data = ds_ic["T"][:, :, :, end]
        S_data = ds_ic["S"][:, :, :, end]

        close(ds_ic)

        # Replace NaNs (land/immersed cells) with 0
        replace!(u_data, NaN => 0.0)
        replace!(v_data, NaN => 0.0)
        replace!(w_data, NaN => 0.0)
        replace!(T_data, NaN => 0.0)
        replace!(S_data, NaN => 0.0)

        set!(model, u=u_data, v=v_data, w=w_data, T=T_data, S=S_data) 
        #=
        using Rasters
        rs = RasterStack(filename, name=(:u, :v, :w, :T, :S))

        @allowscalar u[1:grid.Nx+1, 1:grid.Ny, 1:grid.Nz] .= CuArray(rs.u[ Ti=Near(Inf) ])
        @allowscalar v[1:grid.Nx, 1:grid.Ny, 1:grid.Nz] .= CuArray(rs.v[ Ti=Near(Inf) ])
        @allowscalar w[1:grid.Nx, 1:grid.Ny, 1:grid.Nz+1] .= CuArray(rs.w[ Ti=Near(Inf) ])

        @allowscalar S[1:grid.Nx, 1:grid.Ny, 1:grid.Nz] .= CuArray(rs.S[ Ti=Near(Inf) ])
        @allowscalar T[1:grid.Nx, 1:grid.Ny, 1:grid.Nz] .= CuArray(rs.T[ Ti=Near(Inf) ])
        =#
    else
        @info "Imposing initial conditions from scratch"

        T_ic(x, y, z) = Teast(z, params)

        S_ic(x, y, z) = Seast(x, y, z, params)

        uᵢ = fill(0.0, size(u))
        vᵢ = fill(0.0, size(v))
        wᵢ = fill(0.0, size(w))

        set!(model, T=T_ic, S=S_ic, u=uᵢ, v=vᵢ, w=wᵢ)
    end
    #----

    #++++ Outputs
    @info "Creating output fields"

    #-----------------------------
    #CALCULATIONS FOR OUTPUT FIELDS
    #-----------------------------

    # z-component of vorticity (also known as relative vorticity) calculation; vorticity = ∂x(v) - ∂y(u)
    vorticity_z = Field(∂x(v) - ∂y(u))

    #Kinetic energy calculation for u, v, w
    KE_u = Field(@at (Center, Center, Center) 0.5 * u^2)
    KE_v = Field(@at (Center, Center, Center) 0.5 * v^2)
    KE_w = Field(@at (Center, Center, Center) 0.5 * w^2)
    KE_total = Field(@at (Center, Center, Center) 0.5 * (u^2 + v^2 + w^2))
    
    #Different depths for upwelling outputs (75m, 150m, 225m)
    Δz = params.Lz / params.Nz          # ≈ 16.7m per cell  
    k_75m = Int(floor((params.Lz - 75) / Δz))  # how many cells up from bottom  
    k_150m = Int(floor((params.Lz - 150) / Δz))  # how many cells up from bottom  
    k_225m = Int(floor((params.Lz - 225) / Δz))  # how many cells up from bottom  
    
    #Calculating flux for upwelling at different depths (heavy calculations, so only doing if H_S_flux == 1)

    if H_S_flux == 0
        @info "No heat and salt fluxes being calculated"
    elseif H_S_flux == 1
        @info "Calculating heat and salt fluxes using linear functions of z for temperature and salinity"
        #Background of salinity and temprature are my functions of z from above;
        T_background(x, y, z) = (22-10)*z / 500 + 22
        S_background(x, y, z) = Sₗ(z)

        #Since these are functions, we need to make them into fields to be able to save them as outputs in the netCDF files
        T_background_field = FunctionField((Center, Center, Center), T_background, model.grid)
        S_background_field = FunctionField((Center, Center, Center), S_background, model.grid)

        #Calculate difference of temperature and salinity
        T_diff = Field(T - T_background_field)
        S_diff = Field(S - S_background_field)

        #Flux calculation for upwelling for temperature and salinity; spaital changes
        wT_difference = Field(@at (Center, Center, Center) w * T_diff)
        wS_difference = Field(@at (Center, Center, Center) w * S_diff)

        #Integrated fluxes; scalar
        ∫wT_difference_up = Integral(wT_difference) 
        ∫wS_difference_up = Integral(wS_difference)

        flux_outputs = (; wT_difference, wS_difference, ∫wT_difference_up, ∫wS_difference_up) 
    end
    
    outputs = (; u, v, w, T ,S ,vorticity_z)


    if mass_flux
        saved_output_prefix = "iceplume"
    else
        saved_output_prefix = "iceplume_nomf"
    end
    saved_output_filename = saved_output_prefix * ".nc"
    checkpointer_prefix = "checkpoint_" * saved_output_prefix

    #+++ Check for checkpoints
    if any(startswith("$(checkpointer_prefix)_iteration"), readdir(rundir))
        @warn "Checkpoint $saved_output_prefix found. Assuming this is a pick-up simulation! Setting `overwrite_existing=false`."
        overwrite_existing = true #changed from false
    else
        @warn "No checkpoint for $saved_output_prefix found. Setting `overwrite_existing=true`."
        overwrite_existing = true
    end
    #---

    ccc_scratch = Field{Center, Center, Center}(model.grid) # Create some scratch space to save memory


    if bathymetry_mode == 0
        bathy_tag = "no_bathymetry"
        bathy_folder = "galapagos_netcdf_no_bathymetry"
    elseif bathymetry_mode == 1
        bathy_tag = "gaussian_bathymetry"
        bathy_folder = "galapagos_netcdf_gaussian_bathymetry"
    elseif bathymetry_mode == 2
        bathy_tag = "real_bathymetry"
        bathy_folder = "galapagos_netcdf_real_bathymetry"
    end
    
    if wind == 0   
        windtag = "no_wind"
    elseif wind == 1
        windtag = "wind"
    end

    if beta_switch == 0
        beta_tag = "no_beta"
    elseif beta_switch == 1
        beta_tag = "beta"
    end

    #create a folder for the netCDF outputs if it doesn't already exist
    output_dir = joinpath(rundir, bathy_folder)
    mkpath(output_dir)  





    
    #++++ Save run metadata (parameters + switches + timestamp) to an Excel file
    # This gives every run a self-documenting record: what was swept, when it ran,
    # and which folder/tag its NetCDF outputs live under — so results never get
    # separated from the settings that produced them.
 
    run_timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    run_tag = "$(bathy_tag)_$(windtag)_$(beta_tag)_$(run_timestamp)"
 
    # Anything not already in `params` but useful for identifying/reproducing this run
    switches = Dict(
        "bathymetry_mode"       => bathymetry_mode,
        "bathy_tag"             => bathy_tag,
        "wind"                  => wind,
        "wind_tag"              => windtag,
        "EUC_model"             => EUC_model,
        "beta_switch"           => beta_switch,
        "beta_tag"              => beta_tag,
        "H_S_flux"              => H_S_flux,
        "Smoothing_bathymetry"  => Smoothing_bathymetry,
        "interpolated_IC"       => interpolated_IC,
        "ext_forcing"           => ext_forcing,
        "arch"                  => string(arch),
        "Delta_t0_seconds"      => string(Δt₀),
        "stop_time"             => string(simulation.stop_time),
        "saved_output_prefix"   => saved_output_prefix,
    )
 
    function build_metadata_df(params, switches)
        rows   = String[]
        values = String[]
        for (k, v) in pairs(params)
            push!(rows, String(k))
            push!(values, string(v))   # string() safely handles Unitful quantities too
        end
        for (k, v) in switches
            push!(rows, k)
            push!(values, string(v))
        end
        return DataFrame(Parameter = rows, Value = values)
    end
 
    metadata_df = build_metadata_df(params, switches)
 
    metadata_filename = joinpath(output_dir, "metadata_$(run_tag).xlsx")
 
    XLSX.openxlsx(metadata_filename, mode="w") do xf
        sheet = xf[1]
        XLSX.rename!(sheet, "Metadata")
 
        sheet["A1"] = "Run tag"
        sheet["B1"] = run_tag
        sheet["A2"] = "Timestamp"
        sheet["B2"] = run_timestamp
        sheet["A3"] = "Output folder"
        sheet["B3"] = output_dir
 
        sheet["A5"] = "Parameter"
        sheet["B5"] = "Value"
        for (i, row) in enumerate(eachrow(metadata_df))
            sheet["A$(5+i)"] = row.Parameter
            sheet["B$(5+i)"] = row.Value
        end
    end
 
    @info "Wrote run metadata to $metadata_filename"
    #----

    simulation.output_writers[:surface_slice_writer] =
        NetCDFWriter(model, (; u, v, w, T, S, vorticity_z, KE_u, KE_v, KE_w, KE_total); 
        filename = joinpath(output_dir, "top_$(bathy_tag)_$(windtag)_$(beta_tag)_$(run_timestamp)_GPU.nc"),
                        schedule=TimeInterval(8640seconds), indices=(:, :, params.Nz),
                            overwrite_existing = overwrite_existing)

    #same with below from indicies(:, round (params.NY/2),:)
    simulation.output_writers[:y_slice_writer] =
        NetCDFWriter(model, (; u, v, w, T, S, vorticity_z, KE_u, KE_v, KE_w, KE_total); 
        filename= joinpath(output_dir, "midy_$(bathy_tag)_$(windtag)_$(beta_tag)_$(run_timestamp)_GPU.nc"),
                        schedule=TimeInterval(8640seconds), indices=(:, Int(params.Ny/2), :), 
                        overwrite_existing = overwrite_existing)    


    #Below is the upwelling of different level depths (i.e., k_75m, k_150m, k_225m) and saving those outputs as netCDF files.
    simulation.output_writers[:xy_75_depth_writer] =
        NetCDFWriter(model, (; u, v, w, T, S, vorticity_z, KE_u, KE_v, KE_w, KE_total); 
        filename = joinpath(output_dir, "upwelling_75m_$(bathy_tag)_$(windtag)_$(beta_tag)_$(run_timestamp)_GPU.nc"),
                        schedule=TimeInterval(8640seconds), indices=(:, :, k_75m),
                            overwrite_existing = overwrite_existing)
                            
    simulation.output_writers[:xy_150_depth_writer] =
    NetCDFWriter(model, (; u, v, w, T, S, vorticity_z, KE_u, KE_v, KE_w, KE_total); 
    filename = joinpath(output_dir, "upwelling_150m_$(bathy_tag)_$(windtag)_$(beta_tag)_$(run_timestamp)_GPU.nc"),
                    schedule=TimeInterval(8640seconds), indices=(:, :, k_150m),
                        overwrite_existing = overwrite_existing)
                            
    simulation.output_writers[:xy_225_depth_writer] =
    NetCDFWriter(model, (; u, v, w, T, S, vorticity_z, KE_u, KE_v, KE_w, KE_total); 
    filename = joinpath(output_dir, "upwelling_225m_$(bathy_tag)_$(windtag)_$(beta_tag)_$(run_timestamp)_GPU.nc"),
                    schedule=TimeInterval(8640seconds), indices=(:, :, k_225m),
                        overwrite_existing = overwrite_existing)

    #Save heat and salt fluxes if the switch is on; these are very computationally expensive, so only doing if H_S_flux == 1
    if H_S_flux == 1
        simulation.output_writers[:flux_writer_75] =
            NetCDFWriter(model, (; wT_difference, wS_difference, ∫wT_difference_up, ∫wS_difference_up); 
                        filename = joinpath(output_dir, "fluxes_75m_$(bathy_tag)_$(windtag)_$(beta_tag)_$(run_timestamp)_GPU.nc"),
                        schedule=TimeInterval(8640seconds), indices=(:, :, k_75m),
                         overwrite_existing = overwrite_existing)
        simulation.output_writers[:flux_writer_150] =
            NetCDFWriter(model, (; wT_difference, wS_difference, ∫wT_difference_up, ∫wS_difference_up); 
                        filename = joinpath(output_dir, "fluxes_150m_$(bathy_tag)_$(windtag)_$(beta_tag)_$(run_timestamp)_GPU.nc"),
                        schedule=TimeInterval(8640seconds), indices=(:, :, k_150m),
                         overwrite_existing = overwrite_existing)
        simulation.output_writers[:flux_writer_225] =
            NetCDFWriter(model, (; wT_difference, wS_difference, ∫wT_difference_up, ∫wS_difference_up); 
                        filename = joinpath(output_dir, "fluxes_225m_$(bathy_tag)_$(windtag)_$(beta_tag)_$(run_timestamp)_GPU.nc"),
                        schedule=TimeInterval(8640seconds), indices=(:, :, k_225m),
                         overwrite_existing = overwrite_existing)
    end




    ccc_scratch = Field{Center, Center, Center}(model.grid) # Create some scratch space to save memory

    
    
    # Save a snapshot at the very end for use as IC
    simulation.output_writers[:IC_writer] =
        NetCDFWriter(model, (; u, v, w, T, S, vorticity_z, KE_u, KE_v, KE_w, KE_total);
                    filename = joinpath(output_dir, "IC_$(bathy_tag)_$(windtag)_$(beta_tag)_$(run_timestamp)_GPU.nc"),
                    schedule = TimeInterval(365days),  # only saves at the end
                    overwrite_existing = overwrite_existing)
    
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = TimeInterval(8640seconds),
                                                            prefix = checkpointer_prefix,
                                                            cleanup = true,
                                                            )

    #---
    
    #+++ Ready to press the big red button: 
    run!(simulation; pickup=false) 
    #---