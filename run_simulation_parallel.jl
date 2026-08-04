"""
Structure:
struct RunParameters
    bathymetry_mode::Int   # 0 = No bathymetry, 1 = Gaussian bathymetry, 2 = Real bathymetry
    wind::Int              # 0 = no wind, 1 = wind from 4-year netCDF
    beta_switch::Int       # 0 = no beta, 1 = beta plane
    H_S_flux::Int          # 0 = no flux, 1 = heat/salt flux diagnostics
    smoothing::Int         # 0 = raw bathymetry, 1 = gaussian smoothed
    EUC_model::Int         # 0 = constant forcing, 1 = fourier-based (not yet implemented)
    EUC_value::Float64      # EUC velocity value (m/s)
    model_type::String     # "hydrostatic" or "nonhydrostatic"
end
"""

# Driver: launches N worker processes, runs each RunParameters config in its own
# isolated process (avoids NetCDF/FFTW thread-safety issues), reports failures
# individually instead of one crash taking down the whole batch. After all sims
# finish, each successful run's output is handed to a MATLAB visualization
# script, run sequentially on the main process, and the resulting figure(s)
# are saved into that run's own output folder.

using Distributed
#Change in EUC values
runs_config = [
    (2, 0, 1, 0, 1, 0, 0.5, "nonhydrostatic"),
    (2, 0, 1, 0, 1, 0, 0.1, "nonhydrostatic"),
    (2, 0, 1, 0, 1, 0, 1.0, "nonhydrostatic"),
    (2, 0, 1, 0, 1, 0, 2.0, "nonhydrostatic"),
    #island variation
    (0, 0, 1, 0, 1, 0, 0.5, "nonhydrostatic"),
    (1, 0, 1, 0, 1, 0, 0.5, "nonhydrostatic"),
    #Beta plane variation
    (2, 0, 0, 0, 1, 0, 0.5, "nonhydrostatic"),

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

# ---------------------------------------------------------------------------
# MATLAB visualization pass
# ---------------------------------------------------------------------------
# Runs sequentially on the main process (not through pmap/workers), after all
# simulations are done. This is deliberate: launching N simultaneous MATLAB
# instances can fail on shared HPC systems if you don't have N free MATLAB
# license seats. If you know you have enough seats and want it parallelized,
# swap the `for` loop below for `asyncmap`.

const MATLAB_BIN = "matlab"                          # full path if `matlab` isn't on PATH
const MATLAB_VIZ_FN = "shelfwind_visualization_2"       # <-- your .m function name, no .m extension

"""
    run_matlab_visualization(outdir) -> Symbol

Calls MATLAB in batch mode to generate and save figures for one run's output
directory. Assumes a MATLAB function `MATLAB_VIZ_FN(outdir)` on the MATLAB
path that reads the NetCDF in `outdir` and saves its figure(s) back into that
same folder — swap in whichever of your surface/cross-section/vertical-velocity
plotting functions you want triggered here.

Note: MATLAB calls functions by name, not by filename — pass the name without
a ".m" extension (a trailing one is stripped here just in case). Also, this
assumes MATLAB_VIZ_FN is an actual function (accepts an input argument). If
it's a plain script instead, calling it with `(outdir)` will fail with an
input-arguments error — in that case the script needs a variable set in the
workspace before it runs, rather than being called like a function.
"""
function run_matlab_visualization(outdir::String)
    logfile = joinpath(outdir, "matlab_viz.log")
    fn = replace(MATLAB_VIZ_FN, r"\.m$" => "")
    matlab_stmt = "$(fn)('$(outdir)')"
    # -batch alone is headless on every platform; -nodisplay is Linux/Mac-only
    # and errors out on Windows MATLAB, which is why it's not included here.
    cmd = `$MATLAB_BIN -batch $matlab_stmt`
    open(logfile, "w") do io
        try
            run(pipeline(cmd; stdout=io, stderr=io))
            :success
        catch err
            println(io, "MATLAB visualization failed: $err")
            :failed
        end
    end
end

println("\n===== MATLAB VISUALIZATION =====")
for r in results
    if r.status == :success
        viz_status = run_matlab_visualization(r.output_dir)
        marker = viz_status == :success ? "✓" : "✗"
        println("$marker $(r.params) → $(r.output_dir)/matlab_viz.log")
    end
end

#how to run: julia --project=. run_simulation_parallel.jl