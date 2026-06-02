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
    using Oceananigans.Grids: xnode, ynode, znode
    using Oceanostics.KineticEnergyEquation: KineticEnergy, KineticEnergyStress, DissipationRate
    using Oceanostics.FlowDiagnostics: QVelocityGradientTensorInvariant, RichardsonNumber
    using Oceanostics.TurbulentKineticEnergyEquation: ShearProductionRate, XShearProductionRate, YShearProductionRate, ZShearProductionRate
    using Statistics
    using Printf
    using Oceananigans: Callback, IterationInterval
    using Oceanostics.ProgressMessengers: SingleLineMessenger 
    using CairoMakie

    #-------------------------------------------------------------------------------------
    #CONTROL BOARD
    
    # Bathymetry switch
    bathymetry_mode = 1   # 0 = No bathymetry, 1 = Gaussian bathymetry, 2 = Real bathymetry
    
    #wind switch
    wind = 0

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

    if  bathymetry_mode == 0 
        @info "Using no bathymetry (flat bottom)"
        Lx_real = 100e4
        Ly_real = 500e3
        #Flat bottom
        bottom(x,y) = -500

    elseif bathymetry_mode == 1

        @info "Using gaussian bathymetry"

        #Gaussian Bathymetry of Galapagos
        #height of 500 m, 250km mean, 3e4 (30 km STD)
        Lx_real = 100e4
        Ly_real = 500e3
        bottom(x,y) = -500 + 560 * exp( -(x-params.Lx/2)^2/(2*(3e4)^2) )* exp(-(y-0)^2/(2*(3e4)^2))

    elseif bathymetry_mode == 2
        @info "Using real bathymetry"

        #Load Bathymetry data of the Galapagos
        ds = NCDataset("galap.nc")

        lon = ds["x"][:]   # longitudesThreads.nthreads()
        lat = ds["y"][:]   # latitudes
        zflat = ds["z"][:] # depth values (flattened)
        close(ds)

        nx = length(lon)
        nz = length(lat)

        depth = reshape(zflat, nx, nz) # reshape to 2D grid (lat × lon)


        Lx_real = (maximum(lon) - minimum(lon)) * 111e3
        Ly_real = (maximum(lat) - minimum(lat)) * 111e3

        itp = extrapolate(
            interpolate((lat, lon), collect(depth'), Gridded(Linear())),
            Interpolations.Flat()
        )

        deg_per_meter = 1 / 111e3
        #island_lat = 0
        lon_from_x(x) = minimum(lon) + x * deg_per_meter

        y_offset = 75000  # meters — tune this until island is centered

        lat_from_y(y) = minimum(lat) + (y + Ly_real/2 - y_offset) * deg_per_meter

        bottom(x, y) = itp(lat_from_y(y), lon_from_x(x))

    else
        @warn "Unknown bathymetry_mode; defaulting to real bathymetry"
    end
   


    #++++ Construct grid
    if LES
        params = (; Lx = Lx_real,
                Ly = Ly_real,
                Lz = 500,
                Nx = 50,
                Ny = 50,
                Nz = 50,
                ) 
    end
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


    # Not necessary, but makes organizing simulations easier and facilitates running on GPUs
    params = (; params...,
            N²₀ = 2e-4, #  9.83/1028*2/100  1/s (stratification frequency)
            σ = 40000.0seconds, # s (relaxation timescale for sponge layer) how long we expect it to; now 1 day CHANGedto 40000 seconds (half a day)
            #uₑᵥₐᵣ = 0.00, # m/s (velocity variation along the z direction of the east boundary)
            u_b = 0.0,    # m s⁻¹, average wind velocity 10 meters above the ocean
            v_b = 0.0,    #-10    # m s⁻¹, average wind velocity 10 meters above the ocean
            )
    #----

    #modeling eastward EUC velocity with the data from paper
    const Umaxᵥ = 0.5 #m/s
    const zₒᵥ = -75 #m 
    const yₒᵥ = 0 #m
    const σ_zᵥ = 20 #m #change to around 20 ish
    const σ_yᵥ = 55600 #0.5 * pi/180 * (6.371*10^6) #m = 55,600 meters #divide by two 

    @inline U₁(y) = exp(-(y-yₒᵥ)^2/(2*σ_yᵥ^2))
    @inline U₂(z)= exp(-(z-zₒᵥ)^2/(2*σ_zᵥ^2))
    @inline U(y,z) = Umaxᵥ * U₁(y) * U₂(z)


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
        @inline Uwest(y,z,parameters) = U(y,z)  #m/s
        @inline Ueast(y,z,parameters) = U(y,z) #m/s
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


    model = HydrostaticFreeSurfaceModel(grid, 
                                tracers = (:T, :S),
                                buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = 3.87e-5,
                                haline_contraction = 7.86e-4)), 
                                momentum_advection = WENO(),
                                tracer_advection = WENO(),
                                coriolis = BetaPlane(latitude=0),
                                closure = closure,
                                forcing = forcing,
                                boundary_conditions = boundary_conditions,
                                )

    @info "Model" model



    #----

    #++++ Create simulation
    using Oceanostics: SingleLineProgressMessenger

    Δt₀ = 1/2 * minimum_yspacing(grid) / 1 # / (u₁_west + 1)
    simulation = Simulation(model, Δt=Δt₀,
                            stop_time = 10days, # when to stop the simulation
    )

  
    #++++ Adapt time step
    wizard = TimeStepWizard(cfl=0.8, # How to adjust the time step
                            max_change=1.02, min_change=0.2, min_Δt=0.1seconds) #max_Δt=0.5/√params.N²₀)
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

    # y-component of vorticity
    vorticity_z = Field(∂x(v) - ∂y(u))

    outputs = (; u, v, w, T,S,vorticity_z)

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

    #Kinetic energy calculation for u, v, w
    KE_u = Field(@at (Center, Center, Center) 0.5 * u^2)
    KE_v = Field(@at (Center, Center, Center) 0.5 * v^2)
    KE_w = Field(@at (Center, Center, Center) 0.5 * w^2)
    KE_total = Field(@at (Center, Center, Center) 0.5 * (u^2 + v^2 + w^2))
    
    Δz = params.Lz / params.Nz          # ≈ 16.7m per cell  
    k_75m = Int(round((params.Lz - 75) / Δz))  # how many cells up from bottom  

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

    #create a folder for the netCDF outputs if it doesn't already exist
    output_dir = joinpath(rundir, bathy_folder)
    mkpath(output_dir)  

    simulation.output_writers[:surface_slice_writer] =
        NetCDFWriter(model, (; u, v, w, T, S, vorticity_z, KE_u, KE_v, KE_w, KE_total); 
        filename = joinpath(output_dir, "top_$(bathy_tag)_$(windtag)_GPU.nc"),
                        schedule=TimeInterval(8640seconds), indices=(:, :, params.Nz),
                            overwrite_existing = overwrite_existing)

    #same with below from indicies(:, round (params.NY/2),:)
    simulation.output_writers[:y_slice_writer] =
        NetCDFWriter(model, (; u, v, w, T, S, vorticity_z, KE_u, KE_v, KE_w, KE_total); 
        filename= joinpath(output_dir, "midy_$(bathy_tag)_$(windtag)_GPU.nc"),
                        schedule=TimeInterval(8640seconds), indices=(:, Int(params.Ny/2), :), 
                        overwrite_existing = overwrite_existing)    

    simulation.output_writers[:xy_75_depth_writer] =
        NetCDFWriter(model, (; u, v, w, T, S, vorticity_z, KE_u, KE_v, KE_w, KE_total); 
        filename = joinpath(output_dir, "upwelling_75m_$(bathy_tag)_$(windtag)_GPU.nc"),
                        schedule=TimeInterval(8640seconds), indices=(:, :, k_75m),
                            overwrite_existing = overwrite_existing)
                            
    ccc_scratch = Field{Center, Center, Center}(model.grid) # Create some scratch space to save memory

    
        # Save a snapshot at the very end for use as IC
    simulation.output_writers[:IC_writer] =
        NetCDFWriter(model, (; u, v, w, T, S, vorticity_z, KE_u, KE_v, KE_w, KE_total);
                    filename = joinpath(output_dir, "IC_$(bathy_tag)_$(windtag)_GPU.nc"),
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