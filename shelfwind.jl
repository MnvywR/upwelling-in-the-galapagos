    using Revise
    using Pkg
    #Pkg.instantiate() # Only need to do this once when you started the repo in another machine
    #Pkg.resolve()
    #import Pkg; Pkg.add("Oceananigans")
    #import Pkg; Pkg.add("Rasters")
    #Pkg.instantiate() # Only need to do this once when you started the repo in another machine
    #Pkg.resolve()
    using Oceananigans
    using Oceananigans.Units
    using Printf
    using CUDA: has_cuda_gpu, @allowscalar, CuArray
    using Statistics: mean
    using Oceanostics
    using Rasters
    using NCDatasets #added
    using Interpolations
    using Polynomials
    #added two below
    using Oceanostics
    #added below; some may be redundant, but whatever
    using Oceanostics.KineticEnergyEquation: KineticEnergy, KineticEnergyStress, DissipationRate
    using Oceanostics.FlowDiagnostics: QVelocityGradientTensorInvariant, RichardsonNumber
    using Oceanostics.TurbulentKineticEnergyEquation: ShearProductionRate, XShearProductionRate, YShearProductionRate, ZShearProductionRate
    using Statistics
    using Printf
    using Oceananigans: Callback, IterationInterval
    using Oceanostics.ProgressMessengers: SingleLineMessenger 
    
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
       # arch = GPU()
    #else
        arch = CPU()
    end
    #---

    #It is in units of m
    #Lx,y = 500-600 Km -> 500e3 -> 1000 Km
    #Lz = 500 m 
    #++++ Construct grid
    if LES
        params = (; Lx = 100e4,
                Ly = 500e3,
                Lz = 500,
                Nx = 30,
                Ny = 30,
                Nz = 30,
                ) 

    else
        params = (; Lx = 50000,
                Ly = 100000,
                Lz = 200,
                Nx = 250, #ideally 512
                Ny = 500, ##Int(Nx/2*3/5)
                Nz = 100, #ideally 750
                )
    end
    #changing grid i.e. Nx Ny Nz to 30 30 30 cuz laptop
    if arch == CPU() # If there's no CPU (e.g. if we wanna test shit on a laptop) let's use a smaller number of points!
        params = (; params..., Nx = 30, Ny = 30, Nz = 30)
    end


    #x now periodic, y bounded
    underlying_grid = RectilinearGrid(arch,
    #grid = RectilinearGrid(arch,
                        size = (params.Nx, params.Ny, params.Nz),
                        x = (0, params.Lx),
                        y = (-params.Ly/2, +params.Ly/2),
                        z = (-params.Lz, 0), 
                        halo = (4, 4, 4),
                        topology = (Oceananigans.Grids.Periodic, Oceananigans.Grids.Bounded, Oceananigans.Grids.Bounded))

    #----

    #Bathymetry of Galapagos
    #height of 500 m, 250km mean, 3e4 (30 km STD, w
    bottom(x,y) = -500 + 500 * exp( -(x-params.Lx/2)^2/(2*(3e4)^2) )* exp(-(y-0)^2/(2*(3e4)^2))

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))

    @info "Grid" grid


    # Not necessary, but makes organizing simulations easier and facilitates running on GPUs
    params = (; params...,
            N²₀ = 2e-4, #  9.83/1028*2/100  1/s (stratification frequency)
            σ = 40000.0seconds, # s (relaxation timescale for sponge layer) how long we expect it to; now 1 day CHANGedto 40000 seconds (half a day)
            #uₑᵥₐᵣ = 0.00, # m/s (velocity variation along the z direction of the east boundary)
            u_b = 0,    # m s⁻¹, average wind velocity 10 meters above the ocean
            v_b = 10,    #-10    # m s⁻¹, average wind velocity 10 meters above the ocean
            )
    #----

    #modeling eastward velocity with the data from paper
    Umaxᵥ = 0.5 #m/s
    zₒᵥ = -75 #m 
    yₒᵥ = 0 #m
    σ_zᵥ = 25 #m #change to around 20 ish
    σ_yᵥ = 55600 #0.5 * pi/180 * (6.371*10^6) #m = 55,600 meters #divide by two 

    U₁(y) = exp(-(y-yₒᵥ)^2/(2*σ_yᵥ^2))
    U₂(z)= exp(-(z-zₒᵥ)^2/(2*σ_zᵥ^2))
    U(y,z) = Umaxᵥ * U₁(y) * U₂(z)


    #Modeling temperature profile with the data from paper

    #linear salinity function for the right
    Sₗ(z) = 35.0 + (35.0 - 34.7) * (z / 500)

    #Conditions from Graph
    Smax = 0.6 #psu
    zₒₛ = -75 #m 
    yₒₛ = -1.33e5 #m
    σ_zₛ = 23 #m
    σ_yₛ = 20000 #m

    #Gaussian
    S₁(y) = exp(-(y-yₒₛ)^2/(2*σ_yₛ^2))
    S₂(z)= exp(-(z-zₒₛ)^2/(2*σ_zₛ^2))
    S_anom(y,z) = Smax * S₁(y) * S₂(z)
  

    function S_fxn(x, y, z)
        if y < 0
            return Sₗ(z) + S_anom(y,z)
        else
            return Sₗ(z) 
        end
    end

    #++++ Conditions opposite to the ice wall (@ infinity)
    if LES
    #  b∞(z, parameters) = params.N²₀ * z # Linear stratification in the interior (far from ice face)
    #make it into functions of y
        Teast(z,parameters) = (22-10)*z / 500 + 22 #22 from top to 10 from bottom over 500 meters
        #Seast(z, parameters) = 34 - 1/200 * z   #first number is surface salinity offshore, this is offshore data
        Seast(x,y,z,parameters) = Sₗ(z) #should I add +35 for the background??
        Twest(z,parameters) =  (22-10)*z / 500 + 22 #same deal as Teast
        
        Swest(x,y,z,parameters) = Sₗ(z)
        Uwest(y,z,parameters) = U(y,z)  #m/s
        Ueast(y,z,parameters) = U(y,z) #m/s
        u∞(z, parameters) = @allowscalar u_b
        v∞(z, parameters) = @allowscalar v_b
    end
    #----

    #++++ EAST BCs
    if LES
    #  b_east(y, z, t, p) = b∞(z, p) - 0.02 #2/1028*9.8 #  delta rho/rho0 * g + p.b₁_west / (1 + exp((z-p.z₀)/p.σz_west))
    else
        if mass_flux
            u_west(y, z, t, p) = p.u₁_west # / (1 + exp((z-p.z₀)/p.σz_west))
        end
    end

    #++++ WEST BCs

    if LES
    #  b_west(y, z, t, p) = b∞(z, p) #+ p.b₁_west / (1 + exp((z-p.z₀)/p.σz_west))
    else
        if mass_flux
            u_west(y, z, t, p) = p.u₁_west # / (1 + exp((z-p.z₀)/p.σz_west))
        end
    end


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

        @inline drag_u(x, y, t, u, v, p) = - cᴰ * √(u^2 + v^2) * u
        @inline drag_v(x, y, t, u, v, p) = - cᴰ * √(u^2 + v^2) * v

        drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ=cᴰ,))
        drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:v, :v), parameters=(; cᴰ=cᴰ,))
    end
    #----

    #++++ West sponge layer 
    # (smoothes out the mass flux and gets rid of some of the build up of buoyancy)
    @inline function west_mask(x, y, z)
        x0 = 0  #inner location
        x1 = 50000  #outer location #now 50 km

        if x0 <= x <= x1
            return x/x1
        else
            return 0.0
        end
    end

    #++++ Eastern sponge layer 
    #the shaded function 
    @inline function east_mask(x, y, z)
        x1 = 50000  #bdy width #now 50 km
        x2 = params.Lx-x1  #inner location
        x3 = params.Lx  #outer location

        if x2 <= x <= x3
            return 1 - (params.Lx-x)/x1
        else
            return 0.0
        end
    end

    #@inline sponge_T(x, y, z, t, T, p) = -east_mask(x, y, z) / p.σ * (T - Teast(z, p))-west_mask(x, y, z) / p.σ * (T - (Twest(z, p))) # nudges T to T∞
    
    #Sponge layeres for the temperature, salinity, and eastward velocity 
    @inline sponge_T(x, y, z, t, T, p) = -east_mask(x, y, z) / p.σ * (T - Teast(z, p))-west_mask(x, y, z) / p.σ * (T - (Twest(z, p))) # nudges T to T∞
    @inline sponge_S(x, y, z, t, S, p) = -east_mask(x, y, z) / p.σ * (S - Seast(x, y, z, p))-west_mask(x, y, z) / p.σ * (S - (Swest(x, y, z, p))) # nudges S to S∞
    @inline sponge_u(x, y, z, t, u, p) = -east_mask(x, y, z) / p.σ * (u - Ueast(y, z, p))-west_mask(x, y, z) / p.σ * (u - (Uwest(y, z, p))) # nudges S to S∞

    #----

    #++++ Assembling forcings and BCs
 
    FT = Forcing(sponge_T, field_dependencies = :T, parameters = params)
    FS = Forcing(sponge_S, field_dependencies = :S, parameters = params)
    FU = Forcing(sponge_u, field_dependencies = :u, parameters = params)
    forcing = (T=FT, S=FS, u = FU)

    #Flux boundary conditions -> Value '''' and turned on west and east for Tbcs/Sbcs
    T_bcs = FieldBoundaryConditions(#top = ValueBoundaryCondition(get_T0, field_dependencies=(:T, :S)), 
                                    #west = ValueBoundaryCondition(20), #want to verify if it is Twest, etc. i.e. 20, 35
                                    #east = ValueBoundaryCondition(20), # Hidden behind sponge layer
                                    )

                                    
    S_bcs = FieldBoundaryConditions(#top = ValueBoundaryCondition(get_S0, field_dependencies=(:T, :S)),
                                    #west = ValueBoundaryCondition(35),  
                                    #east = ValueBoundaryCondition(35), # Hidden behind sponge layer
                                    )
    u_bcs = FieldBoundaryConditions(#west = ValueBoundaryCondition(0.3),  
                                    #east = ValueBoundaryCondition(0.3)
                                    #top = FluxBoundaryCondition(Qu),  # wind stress
                                    #bottom = drag_bc_u, # # bottom = drag_bc_u, #must change to quadratic function
                                    )
    v_bcs = FieldBoundaryConditions(#top = FluxBoundaryCondition(Qv),  # wind stress
                                    #bottom = drag_bc_v,
                                # east = ValueBoundaryCondition(0),
                                # west = ValueBoundaryCondition(0),
                                    )
    w_bcs = FieldBoundaryConditions(#east = ValueBoundaryCondition(0),
                                    #west = ValueBoundaryCondition(0),
                                    )
    boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs, T=T_bcs, S=S_bcs,)
    #----

    #++++ Construct model
    if LES
        closure = AnisotropicMinimumDissipation()
        #closure = ScalarDiffusivity(ν=1e-2, κ=1e-2)
    else
        closure = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(),ν=1.8e-6, κ=(T=1.3e-7, S=7.2e-10))
    end

    #θ = 105 # degrees relative to pos. x-axis


    phi = -0.0 * pi / 180 #this is in radians now, did -0.7 degrees for latitude 
    model = HydrostaticFreeSurfaceModel(grid, 
                                #advection = WENO(grid=grid, order=5),
                                #timestepper = :RungeKutta3, 
                                #timestepper =  :RungeKutta3, #:QuasiAdamsBashforth2, 
                                tracers = (:T, :S),
                                buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = 3.87e-5,
                                haline_contraction = 7.86e-4)), 
                                #gravity_unit_vector=(0,0,-1),
                                #tracers = (:T, :S),
                                #buoyancy = BuoyancyTracer(),
                                momentum_advection = WENO(),
                                tracer_advection = WENO(),
                                coriolis = BetaPlane(latitude=phi), #FPlane(1.26e-4),
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
                            stop_time = 100days, # when to stop the simulation
    )

  
    #++++ Adapt time step
    wizard = TimeStepWizard(cfl=0.8, # How to adjust the time step
                        # diffusive_cfl=5,
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

        filename = "IC_part.nc"
        @info "Imposing initial conditions from existing NetCDF file $filename"

        using Rasters
        rs = RasterStack(filename, name=(:u, :v, :w, :T, :S))

        @allowscalar u[1:grid.Nx+1, 1:grid.Ny, 1:grid.Nz] .= CuArray(rs.u[ Ti=Near(Inf) ])
        @allowscalar v[1:grid.Nx, 1:grid.Ny, 1:grid.Nz] .= CuArray(rs.v[ Ti=Near(Inf) ])
        @allowscalar w[1:grid.Nx, 1:grid.Ny, 1:grid.Nz+1] .= CuArray(rs.w[ Ti=Near(Inf) ])

        @allowscalar S[1:grid.Nx, 1:grid.Ny, 1:grid.Nz] .= CuArray(rs.S[ Ti=Near(Inf) ])
        @allowscalar T[1:grid.Nx, 1:grid.Ny, 1:grid.Nz] .= CuArray(rs.T[ Ti=Near(Inf) ])

    else
        @info "Imposing initial conditions from scratch"

        T_ic(x, y, z) = Teast(z, params)

        S_ic(x, y, z) = Seast(x, y, z, params)

    #   b_ic(x, y, z) = b∞(z, params)- 2/1028*9.8*x/params.Lx
        
    #   T_ic(x,y,z) =  2/pi*(T∞(x,params)-T_i)*atan((Lz-z)/delta_B)+T_i

    #    S_ic(x, y, z) = S∞(x, params)
    #   S_ic(x, y, z) = 2/pi*(S∞(x,params)-S_i)*atan((Lz-z)/delta_B*Le^(1/3))+S_i
        
        #initial conditions are uᵢ = 0.005*rand(size(u)...) -> changing to constant:
        #ui =  fill(0.3, size(u)) -> 0.3 m/s
        #i want these initial conditions to be zero i guess
        uᵢ = fill(0.0, size(u))
        vᵢ = fill(0.0, size(v))
        wᵢ = fill(0.0, size(w))

        #uᵢ .-= mean(uᵢ)
        #vᵢ .-= mean(vᵢ)
        #wᵢ .-= mean(wᵢ)
        #uᵢ .+= 0 #params.u_b
        #vᵢ .+= -.25 #-0.2 #params.v_b    

    
    #    plumewidth(x)=.0833*x;
    #    umax=.04
    
    #function u_ic(x, y, z)

    #    if z > Lz-plumewidth(x)
    #        return 5.77*x*umax*(1-(Lz-z)/plumewidth(x))^6*((Lz-z)/plumewidth(x))^(1/2)
    #    else
    #        return 0.0
    #    end
    #end
        
    # uᵢ .+=  u_ic(x,y,z)

        set!(model, T=T_ic, S=S_ic, u=uᵢ, v=vᵢ, w=wᵢ)
    end
    #----

    #++++ Outputs
    @info "Creating output fields"

    # y-component of vorticity
    vorticity_z = Field(∂x(v) - ∂y(u))

    outputs = (; u, v, w, T,S,ω_z)

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
#=

    simulation.output_writers[:surface_slice_writer] =
        NetCDFWriter(model, (; u, v, w, T, S, vorticity_z); filename = joinpath(rundir, "top.nc"),
                        schedule=TimeInterval(8640seconds), indices=(:, :, params.Nz),
                            overwrite_existing = overwrite_existing)
    #same with below from indicies(:, round (params.NY/2),:)
    simulation.output_writers[:y_slice_writer] =
        NetCDFWriter(model, (; u, v, w, T, S, vorticity_z); filename= joinpath(rundir, "midy.nc"),
                        schedule=TimeInterval(8640seconds), indices=(:, Int(params.Ny/2), :), 
                        overwrite_existing = overwrite_existing)    
=#
    ccc_scratch = Field{Center, Center, Center}(model.grid) # Create some scratch space to save memory

    uv = Field((@at (Center, Center, Center) u*v))
    uw = Field((@at (Center, Center, Center) u*w))
    vw = Field((@at (Center, Center, Center) v*w))
    uT = Field((@at (Center, Center, Center) u*T))
    vT = Field((@at (Center, Center, Center) v*T))
    wT = Field((@at (Center, Center, Center) w*T))
    uS = Field((@at (Center, Center, Center) u*S))
    vS = Field((@at (Center, Center, Center) v*S))
    wS = Field((@at (Center, Center, Center) w*S))

    u_yavg = Average(u, dims=(2))
    v_yavg = Average(v, dims=(2))
    w_yavg = Average(w, dims=(2))
    T_yavg = Average(T, dims=(2))
    S_yavg = Average(S, dims=(2))
    uv_yavg = Average(uv, dims=(2))
    uw_yavg = Average(uw, dims=(2))
    vw_yavg = Average(vw, dims=(2))
    uT_yavg = Average(uT, dims=(2))
    vT_yavg = Average(vT, dims=(2))
    wT_yavg = Average(wT, dims=(2))
    uS_yavg = Average(uS, dims=(2))
    vS_yavg = Average(vS, dims=(2))
    wS_yavg = Average(wS, dims=(2))

    KE = KineticEnergy(model)
    ε = DissipationRate(model) 
    εᴰ = KineticEnergyStress(model) 
    ∫KE = Integral(KE)
    ∫ε = Integral(ε)
    ∫εᴰ = Integral(KineticEnergyStress(model)) 

    Q_inv = QVelocityGradientTensorInvariant(model)
    #KE_t = KineticEnergyTendency(model)
    TKE = TurbulentKineticEnergy(model)
    #PR_x = XPressureRedistribution(model)
    SP_x = XShearProductionRate(model; U=Field(u_yavg))
    #PR_y = YPressureRedistribution(model)
    SP_y = YShearProductionRate(model; U=Field(u_yavg))
    #PR_z = ZPressureRedistribution(model)
    SP_z = ZShearProductionRate(model; U=Field(u_yavg))

    ε_yavg = Average(ε, dims=(2))
    KE_yavg = Average(KE, dims=(2))
    εᴰ_yavg = Average(εᴰ, dims=(2))
    #KE_t_yavg = Average(KE_t, dims=(2))
    TKE_yavg = Average(TKE, dims=(2))
    #PR_x_yavg = Average(PR_x, dims=(2))
    SP_x_yavg = Average(SP_x, dims=(2))
    #PR_y_yavg = Average(PR_y, dims=(2))
    SP_y_yavg = Average(SP_y, dims=(2))
    #PR_z_yavg = Average(PR_z, dims=(2))
    SP_z_yavg = Average(SP_z, dims=(2))

    #Kinetic energy calculation for u, v, w
    KE_u = Field(@at (Center, Center, Center) 0.5 * u^2)
    KE_v = Field(@at (Center, Center, Center) 0.5 * v^2)
    KE_w = Field(@at (Center, Center, Center) 0.5 * w^2)

    KE_total = Field(@at (Center, Center, Center) 0.5 * (u^2 + v^2 + w^2))

    KE_output_fields = (; KE_u, KE_v, KE_w, KE_total, KE_yavg, ε_yavg, ∫KE, ∫ε, ∫εᴰ, εᴰ_yavg, TKE_yavg, SP_x_yavg, SP_y_yavg, SP_z_yavg  )

    simulation.output_writers[:surface_slice_writer] =
        NetCDFWriter(model, (; u, v, w, T, S, vorticity_z, KE_u, KE_v, KE_w, KE_total); filename = joinpath(rundir, "top.nc"),
                        schedule=TimeInterval(8640seconds), indices=(:, :, params.Nz),
                            overwrite_existing = overwrite_existing)
    #same with below from indicies(:, round (params.NY/2),:)
    simulation.output_writers[:y_slice_writer] =
        NetCDFWriter(model, (; u, v, w, T, S, vorticity_z, KE_u, KE_v, KE_w, KE_total); filename= joinpath(rundir, "midy.nc"),
                        schedule=TimeInterval(8640seconds), indices=(:, Int(params.Ny/2), :), 
                        overwrite_existing = overwrite_existing)    

    ccc_scratch = Field{Center, Center, Center}(model.grid) # Create some scratch space to save memory


    simulation.output_writers[:nc] = NetCDFWriter(model, KE_output_fields;
                                                        filename = joinpath(rundir, "KE_info.nc"),
                                                        schedule = TimeInterval(8640seconds),
                                                        overwrite_existing = overwrite_existing)
    #=                                                    
    simulation.output_writers[:nQ] = NetCDFWriter(model, (; Q=Q_inv);
                                                        filename = joinpath(rundir, "Q.nc"),
                                                        schedule = TimeInterval(8640seconds),
                                                        overwrite_existing = overwrite_existing)

    =#
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = TimeInterval(8640seconds),
                                                            prefix = checkpointer_prefix,
                                                            cleanup = true,
                                                            )

    #---
    
    #+++ Ready to press the big red button: 
    run!(simulation; pickup=false) 
    #---

