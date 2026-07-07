using Base.Threads

include("galapagos_control_switch_parallel_CPU.jl")


"""
Structure:
    struct RunParameters
        bathymetry_mode::Int # 0 = No bathymetry, 1 = Gaussian bathymetry, 2 = Real bathymetry
        wind::Int #0 for no wind, 1 for wind (using 4 years of data from netCDF file) #NOTHING HERE YETTTTTTTTTTTTTTTTTTTTTTTT
        beta_switch::Int #0 for no beta, 1 for beta plane
        H_S_flux::Int #0 for no flux, 1 for flux (using linear functions of z for temperature and salinity)
        smoothing::Int #0 for original bathymetry, 1 for gaussian smoothed bathymetry
        EUC_model::Int #Look at the function, but 0 = constant forcing, 1 = fourier-based is the data based waveforms #NOTHING HERE YETTTTTTTTTTTTTTTTTTTTTTTT
        model_type::String # "hydrostatic" or "nonhydrostatic"

"""
#Real bathymetry:
runs = [
    RunParameters(2,0,0,0,0,0, "hydrostatic"), 
    RunParameters(2,0,1,0,0,0, "hydrostatic"),
    RunParameters(2,0,1,0,1,0, "hydrostatic")
]

Threads.@threads for i in eachindex(runs)
    println("Starting run $i on thread $(threadid())")

    run_simulation(runs[i])

    println("Finished run $i")
    
end

build_master_metadata(@__DIR__)