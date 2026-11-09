# FLOWFarm Integration in Ard

This folder contains Ard's Python-Julia integration utilities for FLOWFarm.

## Julia setup (required before first use)

FLOWFarm runs inside Julia via [JuliaCall](https://juliapy.github.io/PythonCall.jl/stable/). You need Julia installed before running any FLOWFarm components. Users who do not use FLOWFarm do not need Julia at all — it is loaded lazily only when a FLOWFarm component is initialized.

### 1. Create a conda environment with Python + juliaup (recommended)

```bash
conda create --name ard-FLOWFarm python=3.13 juliaup
conda activate ard-FLOWFarm
```

Set both Julia depots to a path inside the active conda env:

```bash
conda env config vars set JULIA_DEPOT_PATH=${CONDA_PREFIX}/.julia
conda env config vars set JULIAUP_DEPOT_PATH=${CONDA_PREFIX}/.julia
conda deactivate
conda activate ard-FLOWFarm
```

Install Julia **1.11 or newer** using juliaup from that environment (the default `reverse` Jacobian method needs Enzyme, which does not precompile on Julia 1.10; see [Jacobian method](#jacobian-method)):

```bash
juliaup add release
juliaup default release
julia --version
```

Then install Ard (including FLOWFarm dependencies):

```bash
pip install -e ".[dev,flowfarm]"
```

### 2. Pre-generate the Julia environment (optional)

On first use Ard will resolve and instantiate the Julia environment automatically. If you prefer to do this ahead of time — for example on an HPC cluster node without internet access at runtime — run once from your terminal:

```bash
julia --project="<path-to-ard>/ard/farm_aero/flowfarm/julia_env" -e "using Pkg; Pkg.resolve(); Pkg.instantiate()"
```

Replace `<path-to-ard>` with the absolute path to the `Ard` directory. This downloads FLOWFarm and its dependencies. It may take several minutes on first run.

## What this integration does

- Boots Julia through JuliaCall.
- Activates Ard's local Julia environment (`julia_env`).
- Loads FLOWFarm and builds farm and sparse structs for Ard components. Which FLOWFarm is loaded is set by the `julia_env` Manifest (there is no `[sources]` entry); hybridfarm develops it against a FLOWFarm worktree.
- Exposes helper functions used by the component wrapper in `ard/farm_aero/flowfarm/component.py`.

## Threading and parallelism

### OpenMDAO does not multithread

OpenMDAO is single-threaded by design. Its solver loops (Newton, Gauss-Seidel, NLBGS, etc.) are serial. The only parallelism OpenMDAO exposes is MPI-based **process** parallelism via `ParallelGroup`, which spawns separate processes — not threads. This means OpenMDAO will never call the FLOWFarm component from multiple threads simultaneously, so there is no concurrency risk from the OpenMDAO layer.

### Julia internal threading (FLOWFarm parallelism)

Threading in this integration refers to Julia's own thread pool, which FLOWFarm can use internally to parallelize wake calculations across turbines. This is separate from and independent of OpenMDAO.

Julia's thread count is fixed at startup and cannot be changed at runtime. Configure it **before** importing Ard or JuliaCall:

```python
import os
os.environ["PYTHON_JULIACALL_THREADS"] = "4"   # or "auto" to use all cores
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
```

For threaded runs on shared-memory machines, also consider limiting BLAS and OpenMP thread pools to avoid nested oversubscription (Julia threads × BLAS threads × cores):

```python
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
```

These must be set before the first `import ard` call in your script or notebook.

### Why a pure Julia callback

The FLOWFarm update callback in Ard is implemented entirely in Julia (not as a Python callable passed into Julia). This is required for thread safety: JuliaCall does not support calling back into Python from Julia threads other than the main thread. Using a pure Julia callback avoids this restriction and allows FLOWFarm to use all available Julia threads.

## Jacobian method

`modeling_options.flowfarm.jacobian_method` selects how `FLOWFarmAEP` / `FLOWFarmBatchPower` compute gradients:

| value | what it does |
|---|---|
| `reverse` (default) | Per-state Enzyme reverse gradient (`_reverse.jl`), threaded over states. Needs Julia 1.11+. With a `CumulativeCurl` deficit model and FLOWFarm's Enzyme extension (`FLOWFarmEnzymeExt`) loaded, it uses the hand-written `CumulativeCurlAD` rule (about 1.4-2x faster, same gradient to roundoff); `reverse_cc_rule: off` forces plain Enzyme. If the extension is not loaded (e.g. a stale `Manifest.toml`; re-run `Pkg.resolve()`), it falls back to plain Enzyme automatically. |
| `chunked` | Forward chunked ForwardDiff, automatic: dense below `auto_dense_below` turbines (default 25); at or above it a guarded sparse pattern (union of `union_snapshots` = 8 layouts with `union_seed` = 0 and `min_spacing_D` = 3.0) that is self-checked against the dense Jacobian at the first gradient, with dense fallback (RuntimeWarning). Alias: `chunked_auto`. |
| `chunked_dense` | Explicit dense chunked ForwardDiff, no sparsity pruning. |
| `chunked_sparse` | Opt-in color-seeded sparse; cutoffs are fixed when the pattern is built and can be wrong away from the start layout. |
| `di` | FLOWFarm's DifferentiationInterface sparse path (legacy; pruned pattern, loosest agreement). |

`opt_yaw` (default `true`): `false` removes the yaw columns (fewer colors, skips the yaw adjoint); `yaw_turbines` must then be 0, otherwise a ValueError is raised. Discontinuous models (`JensenTopHat`, `LocalTIModelMaxTI`) give unreliable gradients with any method.

On Julia 1.10 set `jacobian_method: chunked` (Enzyme cannot be loaded there). `julia_env/Project.toml` requires Julia 1.11 because Enzyme is a hard dependency of the environment and the default method uses it.

## Tolerance behavior

- FLOWFarm sparse-structure tolerance uses `modeling_options.flowfarm.tolerance`.
- If not provided, the default is `1e-16`.

Example:

```yaml
modeling_options:
  flowfarm:
    tolerance: 1.0e-16
    # jacobian_method: reverse
    # opt_yaw: true
```

## Running the tests

Run the Julia tests with `PYTHON_JULIACALL_HANDLE_SIGNALS=yes` (and `PYTHON_JULIACALL_THREADS` as needed), on Julia 1.11+; otherwise Julia threads can segfault or hang under JuliaCall:

```bash
PYTHON_JULIACALL_HANDLE_SIGNALS=yes pytest -m julia test/flowfarm/integration
```

## Supported FLOWFarm names

Use the exact FLOWFarm constructor and model names expected by the integration.

### Turbine model constructors

- `PowerModelCpPoints`
- `PowerModelConstantCp`
- `ThrustModelCtPoints`
- `ThrustModelConstantCt`

### Wake model options

- `wake_deficit_model`: `JensenTopHat`, `JensenCosine`, `MultiZone`, `GaussOriginal`, `GaussYaw`, `GaussYawVariableSpread`, `GaussSimple`, `CumulativeCurl`, `NoWakeDeficit`
- `wake_deflection_model`: `NoYawDeflection`, `GaussYawDeflection`, `GaussYawVariableSpreadDeflection`, `JiminezYawDeflection`, `MultizoneDeflection`
- `wake_combination_model`: `LinearFreestreamSuperposition`, `SumOfSquaresFreestreamSuperposition`, `SumOfSquaresLocalVelocitySuperposition`, `LinearLocalVelocitySuperposition`
- `local_turbulence_model`: `LocalTIModelNoLocalTI`, `LocalTIModelMaxTI`, `LocalTIModelGaussTI`

## Key files

- `_jl_bootstrap.py`: Julia runtime bootstrap and env activation helpers.
- `flowfarm_model.py`: FLOWFarm model-construction utilities and option validation.
- `component.py`: OpenMDAO component wrapper that uses this integration.
- `_fastjac.jl`: dense and color-seeded sparse chunked ForwardDiff gradients (`chunked*`).
- `_reverse.jl`: per-state Enzyme reverse gradient (`reverse`).

## Troubleshooting

### Julia manifest warnings on first run

If you see warnings like "manifest resolved with a different julia version" or "project dependencies have changed since the manifest was last resolved", it means the local `Manifest.toml` is missing or stale. Ard will attempt to rebuild it automatically. If it does not, run:

```bash
julia --project="<path-to-ard>/ard/farm_aero/flowfarm/julia_env" -e "using Pkg; Pkg.resolve(); Pkg.instantiate()"
```

Then restart your Jupyter kernel. The `Manifest.toml` is not committed to the repository — it is always generated locally for your Julia version.


This comes from your **global** Julia environment, not Ard's. JuliaCall triggers the IPython/Jupyter juliacall extension on import.

### Wrong Julia version being used

Use Julia 1.11 or newer for the default `reverse` method. If an older Julia (e.g. 1.10) is picked up, check your `PATH`: `juliaup default release` sets the default for commands run via juliaup, but if `/opt/homebrew/bin/julia` or another system Julia takes precedence in your shell, JuliaCall may use that instead. On Julia 1.10 set `jacobian_method: chunked`.

### Kernel/process crash when threads > 1

- Ensure pure Julia callback path is active (current Ard default).
- Ensure thread env vars are set before importing Ard.
- Start with `PYTHON_JULIACALL_THREADS=1`, then increase.