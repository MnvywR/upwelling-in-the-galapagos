"""
Structure:
    struct RunParameters
        bathymetry_mode::Int # 0 = No bathymetry, 1 = Gaussian bathymetry, 2 = Real bathymetry
        wind::Int #0 for no wind, 1 for wind (using 4 years of data from netCDF file)
        beta_switch::Int #0 for no beta, 1 for beta plane
        H_S_flux::Int #0 for no flux, 1 for flux (using linear functions of z for temperature and salinity)
        smoothing::Int #0 for original bathymetry, 1 for gaussian smoothed bathymetry
        EUC_model::Int #0 = constant forcing, 1 = fourier-based
        model_type::String # "hydrostatic" or "nonhydrostatic"
"""
# Driver: launches N worker processes, runs each RunParameters config in its own
# isolated process (avoids NetCDF/FFTW thread-safety issues), reports failures
# individually instead of one crash taking down the whole batch.

using Distributed

runs_config = [
    (2, 0, 0, 0, 0, 0, "hydrostatic"),
    (2, 0, 1, 0, 0, 0, "hydrostatic"),
    (2, 0, 1, 0, 1, 0, "hydrostatic"),
]

n_workers = length(runs_config)
addprocs(n_workers; exeflags="--project=. -t4")   # tune -t4 to (cores / n_workers)

@everywhere include("galapagos_control_switch_parallel_CPU.jl")

runs = [RunParameters(cfg...) for cfg in runs_config]

results = pmap(runs) do params
    try
        outdir = run_simulation(params)
        (params=params, status=:success, output_dir=outdir, error=nothing)
    catch err
        bt = catch_backtrace()
        (params=params, status=:failed, output_dir=nothing, error=(err, bt))
    end
end

println("\n===== RUN SUMMARY =====")
for r in results
    if r.status == :success
        println("✓ SUCCESS: $(r.params) → $(r.output_dir)")
    else
        println("✗ FAILED:  $(r.params)")
        err, bt = r.error
        showerror(stderr, err, bt)
        println(stderr)
    end
end

build_master_metadata(@__DIR__)

#how to run: julia --project=. run_simulation_parallel.jl