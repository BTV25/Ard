from pathlib import Path

import numpy as np

from ..floris import create_FLORIS_turbine_from_windIO
from .flowfarm_model import (
    ensure_flowfarm_loaded,
    resolve_turbine_inputs_for_flowfarm,
    resolve_wake_model_inputs_for_flowfarm,
    to_julia_vector_float64,
)

from .. import templates

_PRIMAL_HELPER_DEFINED = False

# Primal per-state turbine powers (W), threaded over states; same calls as
# FLOWFarm.calculate_state_aep, which is sum(powers) * hours_per_year * prob.
_PRIMAL_HELPER = """
function ard_state_turbine_powers(FF, farm)
    c = farm.constants
    wr = c.wind_resource
    nstates = length(wr.wind_probabilities)
    out = zeros(nstates, length(farm.turbine_x))
    Threads.@threads for i = 1:nstates
        rot_x, rot_y = FF.rotate_to_wind_direction(farm.turbine_x, farm.turbine_y, wr.wind_directions[i])
        sorted_turbine_index = sortperm(rot_x)
        v = FF.turbine_velocities_one_direction_vel(rot_x, rot_y, c.turbine_z, farm.rotor_diameter,
                farm.hub_height, farm.turbine_yaw, sorted_turbine_index, c.ct_models,
                c.rotor_sample_points_y, c.rotor_sample_points_z, wr, c.model_set; wind_farm_state_id=i)
        out[i, :] = FF.turbine_powers_one_direction(c.generator_efficiency, c.cut_in_speed,
                c.cut_out_speed, c.rated_speed, c.rated_power, farm.rotor_diameter, v,
                farm.turbine_yaw, wr.air_density, c.power_models)
    end
    return out
end
"""


# modeling_options.flowfarm keys consumed by Ard itself (not passed to FLOWFarm's wake model resolution)
_ARD_OPTIONS = (
    "jacobian_method",
    "auto_dense_below",
    "union_snapshots",
    "union_seed",
    "min_spacing_D",
)


class FLOWFarmComponent:

    opt_yaw = True
    sparse_struct = None
    # Test hook only (not a user option): scales the chunked-auto sparsity cutoffs to force the dense fallback.
    _test_threshold_scale = 1.0

    def initialize(self):
        # This mixin is invoked explicitly by derived classes; no super() chain here.
        return

    def _get_air_density(self, wind_resource):
        return float(wind_resource.get("air_density", 1.225))

    def _get_wake_model_options(self, model_options):
        opts = dict(model_options.get("flowfarm", {}) or {})
        # opt_yaw: False drops the yaw columns from the sparse Jacobian (fewer colors);
        # yaw partials are not declared and the gradient requires yaw_turbines == 0
        # (the forward pass still honors yaw).
        self.opt_yaw = bool(opts.pop("opt_yaw", True))
        # Ard-side options (not wake model inputs): read directly from model_options elsewhere.
        for key in _ARD_OPTIONS:
            opts.pop(key, None)
        return resolve_wake_model_inputs_for_flowfarm(opts)

    def _build_wind_resource(
        self,
        jl,
        flowfarm_module,
        windrose_floris,
        ref_height,
        ref_air_density,
        wind_shear,
    ):
        if hasattr(windrose_floris, "wd_flat"):
            # WindRose (probability-based)
            dirs = np.deg2rad(np.asarray(windrose_floris.wd_flat))
            speeds = np.asarray(windrose_floris.ws_flat)
            probs = np.asarray(windrose_floris.freq_table_flat)
            n_states = len(speeds)
            mean_ti = float(np.mean(windrose_floris.ti_table_flat))
        else:
            # TimeSeries (temporal dispatch)
            dirs = np.deg2rad(np.asarray(windrose_floris.wind_directions))
            speeds = np.asarray(windrose_floris.wind_speeds)
            n_states = len(speeds)
            probs = np.full(n_states, 1.0 / n_states)
            mean_ti = float(np.mean(windrose_floris.turbulence_intensities))

        wind_dirs_rad = to_julia_vector_float64(jl, dirs)
        wind_speeds_vec = to_julia_vector_float64(jl, speeds)
        wind_probs_vec = to_julia_vector_float64(jl, probs)
        ambient_tis = jl.fill(mean_ti, n_states)
        measurementheight = jl.fill(float(ref_height), n_states)
        wind_shear_model = flowfarm_module.PowerLawWindShear(float(wind_shear))

        return flowfarm_module.DiscretizedWindResource(
            wind_dirs_rad,
            wind_speeds_vec,
            wind_probs_vec,
            measurementheight,
            float(ref_air_density),
            ambient_tis,
            wind_shear_model,
        )

    def _build_wake_model_set(self, flowfarm_module, wake_model_options):
        wake_deficit = getattr(
            flowfarm_module, wake_model_options["wake_deficit_model"]
        )()
        wake_deflection = getattr(
            flowfarm_module, wake_model_options["wake_deflection_model"]
        )()
        wake_combine = getattr(
            flowfarm_module, wake_model_options["wake_combination_model"]
        )()
        local_ti = getattr(
            flowfarm_module, wake_model_options["local_turbulence_model"]
        )()

        return flowfarm_module.WindFarmModelSet(
            wake_deficit,
            wake_deflection,
            wake_combine,
            local_ti,
        )

    def _create_update_fn(self, jl):
        jl.seval(
            """
            function ard_make_flowfarm_update_fn()
                return function (farm, x)
                    n = length(farm.turbine_x)
                    @inbounds for i in 1:n
                        farm.turbine_x[i] = x[i]
                        farm.turbine_y[i] = x[n + i]
                        length(x) >= 3n && (farm.turbine_yaw[i] = x[2n + i])
                    end
                    return nothing
                end
            end
            """
        )
        return jl.ard_make_flowfarm_update_fn()

    def _build_farm_structures(
        self,
        jl,
        flowfarm_module,
        N_turbines,
        hub_height,
        rotor_diameter,
        generator_efficiency,
        cutin_wind_speed,
        cutout_wind_speed,
        rated_wind_speed,
        rated_power,
        windresource,
        ct_models,
        power_models,
        model_set,
        tolerance,
        x_init=None,
        y_init=None,
    ):
        # Use actual initial positions if provided so the sparsity pattern
        # computed by build_unstable_sparse_struct sees a non-degenerate farm
        # geometry (x0=zeros puts all turbines at the origin, hiding x/y deps).
        if x_init is None:
            x_init = np.zeros(N_turbines)
        if y_init is None:
            y_init = np.zeros(N_turbines)
        x0_np = np.concatenate([x_init, y_init, np.zeros(N_turbines)])
        x0 = to_julia_vector_float64(jl, x0_np)
        turbine_x = to_julia_vector_float64(jl, x_init)
        turbine_y = to_julia_vector_float64(jl, y_init)
        turbine_z = jl.zeros(N_turbines)
        turbine_yaw = jl.zeros(N_turbines)

        hub_heights = jl.fill(float(hub_height), N_turbines)
        rotor_diameters = jl.fill(float(rotor_diameter), N_turbines)
        generator_efficiencies = jl.fill(float(generator_efficiency), N_turbines)
        cut_in_speeds = jl.fill(float(cutin_wind_speed), N_turbines)
        cut_out_speeds = jl.fill(float(cutout_wind_speed), N_turbines)
        rated_speeds = jl.fill(float(rated_wind_speed), N_turbines)
        rated_powers = jl.fill(float(rated_power), N_turbines)
        update_fn = self._create_update_fn(jl)
        # positional arguments shared by every FLOWFarm farm-struct builder below (after x0)
        common = (
            turbine_x, turbine_y, turbine_z, hub_heights, turbine_yaw, rotor_diameters, ct_models,
            generator_efficiencies, cut_in_speeds, cut_out_speeds, rated_speeds, rated_powers,
            windresource, power_models, model_set, update_fn,
        )

        # x is [x; y] (2N) when yaw is not optimized; update_fn then leaves yaw at 0.
        x0_sparse = x0 if self.opt_yaw else to_julia_vector_float64(jl, x0_np[: 2 * N_turbines])

        # Built lazily on the first gradient request: primal-only runs never pay it.
        self._build_sparse = lambda: flowfarm_module.build_unstable_sparse_struct(
            x0_sparse,
            *common,
            AEP_scale=1,
            opt_x=True,
            opt_y=True,
            opt_yaw=self.opt_yaw,
            tolerance=tolerance,
        )

        self._tolerance = tolerance
        # ForwardDiffJacobian farm used by FLOWFarm to compute the sparsity cutoffs.
        self._build_fd_farm = lambda: flowfarm_module.build_wind_farm_struct(
            x0_sparse,
            *common,
            AEP_scale=1,
            opt_x=True,
            opt_y=True,
            opt_yaw=self.opt_yaw,
            input_type="ForwardDiffJacobian",
        )
        self._x_start = x_init
        self._y_start = y_init
        self._rotor_diameter = rotor_diameter
        # Dense chunked ForwardDiff farm: preallocations dual-typed with chunk width w.
        self._build_dense_farm = lambda w: flowfarm_module.build_wind_farm_struct(
            x0_sparse,
            *common,
            AEP_scale=1,
            opt_x=True,
            opt_y=True,
            opt_yaw=self.opt_yaw,
            input_type=jl.ArdFastJac.dual_type(w),
        )

        farm = flowfarm_module.build_wind_farm_struct(
            x0,
            *common,
            AEP_scale=1,
        )

        return x0, farm

    def _initial_turbine_positions(self, model_options, rotor_diameter):
        """Compute a non-degenerate calibration layout for sparsity detection.

        build_unstable_sparse_struct computes the sparse Jacobian pattern --
        and the per-wind-state deficit threshold used to gate it -- by
        perturbing the design vector at this layout, then reuses that
        pattern/threshold for every later compute_partials() call, including
        calls at the real (possibly very different, optimizer-driven) layout.
        If the calibration layout is geometrically unrelated to the real one,
        near-cut-in wind states can calibrate to an exactly-zero jacobian
        (e.g. a regular grid's unwaked front row plus a uniformly wake-clipped
        rear) and get threshold=Inf, permanently zeroing real, nonzero x/y
        sensitivity at the real layout for that wind state.

        We therefore jitter the actual config-specified layout (config
        layout.x_turbines/y_turbines) by a small random offset -- close enough
        to the real layout that the calibration is representative, but
        perturbed enough to avoid the exact-symmetry degeneracies (e.g.
        turbines sitting on an identical line relative to the wind direction)
        that produce an exactly-zero calibration jacobian. Verified against
        FLOWFarm directly (FarmSetup/test_sparse_threshold_bug.jl) to match
        the true dense gradient to machine precision across 36 wind states
        spanning the full cut-in-to-cut-out envelope, vs. an exact-zero
        sparse gradient at 27+ near-cut-in states with the old synthetic,
        rotated-grid calibration.

        Falls back to a synthetic rotated grid (the previous behavior) only
        if the config has no explicit x_turbines/y_turbines to jitter around.
        """
        layout = model_options.get("layout", {})
        N = self.N_turbines
        x_real = layout.get("x_turbines")
        y_real = layout.get("y_turbines")

        if x_real is not None and y_real is not None and len(x_real) == N and len(y_real) == N:
            rng = np.random.default_rng(0)
            x_jit = np.asarray(x_real, dtype=float) + rng.uniform(0.0, rotor_diameter, size=N)
            y_jit = np.asarray(y_real, dtype=float) + rng.uniform(0.0, rotor_diameter, size=N)
            return x_jit, y_jit

        # Fallback: no explicit layout coordinates in config (e.g. a pure
        # grid-parameterized optimization with no starting x/y) -- synthesize
        # a spread-out grid so x/y position changes aren't degenerate at
        # x0=0 (all turbines collocated).
        spacing_prim = float(layout.get("spacing_primary", 7.0))
        spacing_sec = float(layout.get("spacing_secondary", spacing_prim))
        N_sq = int(np.sqrt(N))

        cy, cx = np.meshgrid(
            np.arange(-((N_sq - 1) / 2), ((N_sq + 1) / 2)),
            np.arange(-((N_sq - 1) / 2), ((N_sq + 1) / 2)),
        )
        if N == N_sq ** 2:
            pass
        elif N <= N_sq * (N_sq + 1):
            # N is between N_sq² and N_sq*(N_sq+1): append a trailing row
            cx = np.vstack([cx, ((N_sq + 1) / 2) * np.ones((N_sq,))])
            cy = np.vstack([cy, np.arange(-((N_sq - 1) / 2), ((N_sq + 1) / 2))])
        else:
            # N is close to (N_sq+1)²: use a wider grid
            cy, cx = np.meshgrid(
                np.arange(-((N_sq) / 2), ((N_sq + 2) / 2)),
                np.arange(-((N_sq) / 2), ((N_sq + 2) / 2)),
            )

        x_init = (cx.ravel()[:N] * spacing_prim * rotor_diameter).astype(float)
        y_init = (cy.ravel()[:N] * spacing_sec * rotor_diameter).astype(float)

        # Rotate the grid by a non-round angle so no turbines are axis-aligned
        # with the predominant wind directions. A regular grid has columns exactly
        # along 0°/90°, so threshold = min(large_deficit)*0.1 ends up too large and
        # excludes weakly-waking pairs at other layouts.
        theta = np.radians(20.0)
        c, s = np.cos(theta), np.sin(theta)
        x_rot = c * x_init - s * y_init
        y_rot = s * x_init + c * y_init
        return x_rot, y_rot

    def setup(self):
        jl = ensure_flowfarm_loaded()
        self._jl = jl
        model_options = self.options["modeling_options"]
        self.N_turbines = model_options["layout"]["N_turbines"]
        windIO = model_options["windIO_plant"]
        wind_resource = windIO["site"]["energy_resource"]["wind_resource"]

        turbine_floris = create_FLORIS_turbine_from_windIO(windIO)
        ref_air_density = self._get_air_density(wind_resource)

        hub_height = turbine_floris["hub_height"]
        rotor_diameter = turbine_floris["rotor_diameter"]

        windIOturbine = windIO["wind_farm"]["turbine"]
        turbine_inputs = resolve_turbine_inputs_for_flowfarm(windIOturbine)
        generator_efficiency = turbine_inputs["generator_efficiency"]
        rated_power = turbine_inputs["rated_power"]
        rated_wind_speed = turbine_inputs["rated_wind_speed"]
        cutin_wind_speed = turbine_inputs["cutin_wind_speed"]
        cutout_wind_speed = turbine_inputs["cutout_wind_speed"]
        ct_model = turbine_inputs["ct_model"]
        power_model = turbine_inputs["power_model"]

        wind_resource_dict = windIO["site"]["energy_resource"]["wind_resource"]
        resource_type = "timeseries" if "time" in wind_resource_dict else "probability"
        windrose_floris = templates.create_windresource_from_windIO(
            windIO,
            resource_type=resource_type,
        )

        ref_height = wind_resource.get("reference_height", hub_height)
        wind_shear = wind_resource.get("shear", 0.084)

        wake_model_options = self._get_wake_model_options(model_options)

        # FLOWFarm expects one model object per turbine.
        ct_models = jl.fill(ct_model, self.N_turbines)
        power_models = jl.fill(power_model, self.N_turbines)

        flowfarm_module = jl.FLOWFarm
        windresource = self._build_wind_resource(
            jl,
            flowfarm_module,
            windrose_floris,
            ref_height,
            ref_air_density,
            wind_shear,
        )
        model_set = self._build_wake_model_set(flowfarm_module, wake_model_options)

        x_init, y_init = self._initial_turbine_positions(model_options, rotor_diameter)
        # flowfarm.jacobian_method: "reverse" (default; per-state Enzyme reverse, _reverse.jl,
        # FLOWFarmBatchPower and FLOWFarmAEP; the AEP component sums the per-state rows),
        # "chunked" (forward: dense below auto_dense_below turbines, else a union-of-layouts sparse
        # pattern self-checked against dense with dense fallback; alias "chunked_auto"),
        # "chunked_dense" (explicit dense chunked ForwardDiff, no sparsity pruning, _fastjac.jl),
        # "chunked_sparse" (opt-in: color-seeded chunked ForwardDiff on FLOWFarm's
        # sparse pattern, whose cutoffs are set when the pattern is built and can be off) or
        # "di" (FLOWFarm's DifferentiationInterface sparse path). Discontinuous models
        # (JensenTopHat, LocalTIModelMaxTI) give unreliable gradients with any method.
        method = model_options.get("flowfarm", {}).get("jacobian_method", "reverse")
        if method not in ("reverse", "chunked", "chunked_auto", "chunked_dense", "chunked_sparse", "di"):
            raise ValueError(
                "flowfarm.jacobian_method must be 'reverse', 'chunked' (auto; alias 'chunked_auto'), "
                f"'chunked_dense', 'chunked_sparse' or 'di', got {method!r}"
            )
        ff_options = model_options.get("flowfarm", {})
        if method in ("chunked", "chunked_auto"):
            # dense for small farms; above the threshold a union-of-layouts sparse pattern that is
            # self-checked against the dense Jacobian at the first gradient (dense on failure)
            if self.N_turbines < int(ff_options.get("auto_dense_below", 25)):
                method = "chunked_dense"
            else:
                method = "chunked_sparse"
                self._auto_pending = True
        x0, farm = self._build_farm_structures(
            jl,
            flowfarm_module,
            self.N_turbines,
            hub_height,
            rotor_diameter,
            generator_efficiency,
            cutin_wind_speed,
            cutout_wind_speed,
            rated_wind_speed,
            rated_power,
            windresource,
            ct_models,
            power_models,
            model_set,
            wake_model_options.get("tolerance", 1e-16),
            x_init=x_init,
            y_init=y_init,
        )

        self.flowfarm_module = flowfarm_module
        self.x0 = x0
        self.farm = farm
        self.sparse_farm = None
        self.sparse_struct = None

        self._fastjac_cache = None
        if method in ("chunked_dense", "chunked_sparse") and not jl.seval("isdefined(Main, :ArdFastJac)"):
            jl.seval(Path(__file__).with_name("_fastjac.jl").read_text())
        if method == "reverse":
            if not jl.seval("isdefined(Main, :ArdReverse)"):
                jl.seval(Path(__file__).with_name("_reverse.jl").read_text())
            n_states = len(np.asarray(farm.constants.wind_resource.wind_speeds).ravel())
            self.sparse_farm = farm
            # opt_yaw False (yaw not a design variable, must stay 0): skips the yaw adjoint and
            # the (identically zero) yaw deflection; yaw columns of state_gradients are then 0.
            self.sparse_struct = jl.ArdReverse.ReverseState(
                farm, n_states, self.N_turbines, 3 * self.N_turbines, opt_yaw=self.opt_yaw
            )
            self._reverse = True
        elif method == "chunked_dense":
            n_states = len(np.asarray(farm.constants.wind_resource.wind_speeds).ravel())
            ndv = (3 if self.opt_yaw else 2) * self.N_turbines
            w = min(8, ndv)
            self.sparse_farm = self._build_dense_farm(w)
            self.sparse_struct = jl.ArdFastJac.DenseState(self.sparse_farm, n_states, self.N_turbines, ndv, w)
            self._dense = True
        elif method == "chunked_sparse":
            # sparse struct is built lazily; take the state count from the wind resource
            n_states = len(np.asarray(farm.constants.wind_resource.wind_speeds).ravel())
            self._fastjac_cache = jl.ArdFastJac.FastCache(n_states)

    def _build_design_vector(self, inputs):
        x_turbines = np.asarray(inputs["x_turbines"], dtype=float)
        y_turbines = np.asarray(inputs["y_turbines"], dtype=float)
        yaw_turbines = np.deg2rad(np.asarray(inputs["yaw_turbines"], dtype=float))
        return np.concatenate([x_turbines, y_turbines, yaw_turbines]).ravel()

    def _check_yaw(self, x_eval_np):
        if not self.opt_yaw and np.any(x_eval_np[2 * self.N_turbines :] != 0.0):
            raise ValueError(
                "FLOWFarm opt_yaw is False but yaw_turbines is nonzero; set opt_yaw: true."
            )

    def _union_layouts(self, K, seed, min_spacing_D):
        """K seeded random layouts inside the convex hull of the start layout, spacing >= min_spacing_D."""
        from scipy.spatial import Delaunay

        N = self.N_turbines
        pts = np.column_stack([self._x_start, self._y_start])
        D = self._rotor_diameter
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        hull = Delaunay(pts) if N >= 3 else None
        d0 = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
        d0 = d0[np.triu_indices(N, 1)].min() if N > 1 else np.inf
        rng = np.random.default_rng(seed)
        layouts = []
        for _ in range(K):
            smin = min(min_spacing_D * D, d0)  # never ask for more than the start layout has
            while True:
                placed = []
                for _try in range(200 * N):
                    q = rng.uniform(lo, hi)
                    if hull is not None and hull.find_simplex(q) < 0:
                        continue
                    if placed and np.min(np.linalg.norm(np.asarray(placed) - q, axis=1)) < smin:
                        continue
                    placed.append(q)
                    if len(placed) == N:
                        break
                if len(placed) == N:
                    break
                smin *= 0.9
            layouts.append(np.asarray(placed))
        return layouts

    def _finish_auto_setup(self, jl):
        """chunked (auto) above the dense threshold: union cutoffs, self-check, dense fallback."""
        import warnings

        self._auto_pending = False
        opts = self.options["modeling_options"].get("flowfarm", {})
        K = int(opts.get("union_snapshots", 8))
        N = self.N_turbines
        self.sparse_farm, self.sparse_struct = self._build_sparse()
        self._fastjac_cache = jl.ArdFastJac.FastCache(
            len(np.asarray(self.farm.constants.wind_resource.wind_speeds).ravel())
        )

        def design(xy):
            v = np.concatenate([xy[:, 0], xy[:, 1]])
            return np.concatenate([v, np.zeros(N)]) if self.opt_yaw else v

        x0 = to_julia_vector_float64(jl, design(np.column_stack([self._x_start, self._y_start])))
        self.auto_info = {"density_single": float(jl.ArdFastJac.pair_density(self.sparse_struct, x0))}
        if K > 0:
            farm_fd = self._build_fd_farm()
            for lay in self._union_layouts(K, int(opts.get("union_seed", 0)), float(opts.get("min_spacing_D", 3.0))):
                getattr(jl.ArdFastJac, "union_snapshot!")(
                    self.sparse_struct, farm_fd, to_julia_vector_float64(jl, design(lay)), self._tolerance
                )
        if self._test_threshold_scale != 1.0:
            getattr(jl.ArdFastJac, "scale_thresholds!")(self.sparse_struct, float(self._test_threshold_scale))
        self.auto_info["density_union"] = float(jl.ArdFastJac.pair_density(self.sparse_struct, x0))

        n_states = len(np.asarray(self.farm.constants.wind_resource.wind_speeds).ravel())
        ndv = (3 if self.opt_yaw else 2) * N
        w = min(8, ndv)
        dense_farm = self._build_dense_farm(w)
        dense = jl.ArdFastJac.DenseState(dense_farm, n_states, N, ndv, w)
        worst, nbad = jl.ArdFastJac.selfcheck(self.sparse_struct, self._fastjac_cache, dense, x0, 8, 1e-8)
        self.auto_info.update(selfcheck_rel=float(worst), selfcheck_bad_pairs=int(nbad))
        if worst > 1e-8:
            warnings.warn(
                f"FLOWFarm chunked: sparse Jacobian differs from dense by {worst:.2e} (relative) "
                f"at the start layout ({int(nbad)} bad turbine pairs over the checked states); "
                "falling back to dense chunked ForwardDiff.",
                RuntimeWarning,
            )
            self._fastjac_cache = None
            self.sparse_farm = dense_farm
            self.sparse_struct = dense
            self._dense = True
            self.auto_info["fallback"] = True
        else:
            self.auto_info["fallback"] = False

    def _evaluate_sparse(self, x_eval_np):
        """Run sparse gradient evaluation once and cache AEP/gradient for reuse."""
        if hasattr(self, "_cached_sparse_x") and np.array_equal(
            self._cached_sparse_x, x_eval_np
        ):
            return

        jl = getattr(self, "_jl", None)
        if jl is None:
            jl = ensure_flowfarm_loaded()
            self._jl = jl
        self._check_yaw(x_eval_np)
        if getattr(self, "_reverse", False):
            # ReverseState takes the full [x; y; yaw] vector (yaw columns 0 when opt_yaw is False).
            aep_val, grad_val = getattr(jl.ArdReverse, "gradient!")(
                self.sparse_farm, to_julia_vector_float64(jl, x_eval_np), self.sparse_struct
            )
        elif getattr(self, "_auto_pending", False):
            self._finish_auto_setup(jl)
            return self._evaluate_sparse(x_eval_np)
        elif getattr(self, "_dense", False):
            x_dense = x_eval_np if self.opt_yaw else x_eval_np[: 2 * self.N_turbines]
            aep_val, grad_val = getattr(jl.ArdFastJac, "dense_gradient!")(
                self.sparse_farm, to_julia_vector_float64(jl, x_dense), self.sparse_struct
            )
        else:
            if self.sparse_struct is None:
                self.sparse_farm, self.sparse_struct = self._build_sparse()
            x_sparse = x_eval_np if self.opt_yaw else x_eval_np[: 2 * self.N_turbines]
            x_eval = to_julia_vector_float64(jl, x_sparse)
            if getattr(self, "_fastjac_cache", None) is not None:
                aep_val, grad_val = getattr(jl.ArdFastJac, "gradient!")(
                    self.sparse_farm, x_eval, self.sparse_struct, self._fastjac_cache
                )
            else:
                calculate_grad_bang = getattr(self.flowfarm_module, "calculate_aep_gradient!")
                aep_val, grad_val = calculate_grad_bang(
                    self.sparse_farm,
                    x_eval,
                    self.sparse_struct,
                )

        self._cached_sparse_x = x_eval_np.copy()
        self._cached_sparse_aep = float(np.asarray(aep_val).ravel()[0])
        self._cached_sparse_grad = np.asarray(grad_val).ravel().copy()

    def _evaluate_farm(self, x_eval_np):
        """Run regular farm AEP evaluation and cache AEP."""
        if hasattr(self, "_cached_farm_x") and np.array_equal(
            self._cached_farm_x, x_eval_np
        ):
            return

        jl = getattr(self, "_jl", None)
        if jl is None:
            jl = ensure_flowfarm_loaded()
            self._jl = jl
        x_eval = to_julia_vector_float64(jl, x_eval_np)
        calculate_aep_bang = getattr(self.flowfarm_module, "calculate_aep!")
        aep_val = calculate_aep_bang(self.farm, x_eval)

        self._cached_farm_x = x_eval_np.copy()
        self._cached_farm_aep = float(np.asarray(aep_val).ravel()[0])

    def _compute_aep(self, inputs, outputs):
        """Compute farm AEP using regular calculate_aep!(farm, x)."""
        x_eval_np = self._build_design_vector(inputs)
        self._evaluate_farm(x_eval_np)
        outputs["AEP_farm"] = self._cached_farm_aep

    def _compute_aep_partials(self, inputs, partials):
        """Compute AEP partial derivatives from sparse gradient evaluation."""
        x_eval_np = self._build_design_vector(inputs)
        self._evaluate_sparse(x_eval_np)
        grad = self._cached_sparse_grad
        partials["AEP_farm", "x_turbines"] = grad[: self.N_turbines]
        partials["AEP_farm", "y_turbines"] = grad[self.N_turbines : 2 * self.N_turbines]
        if self.opt_yaw:
            # Julia gradient is d/d(yaw_rad); yaw_turbines input is in degrees → chain rule factor
            partials["AEP_farm", "yaw_turbines"] = (
                grad[2 * self.N_turbines : 3 * self.N_turbines] * np.deg2rad(1.0)
            )


class FLOWFarmAEP(templates.FarmAEPTemplate, FLOWFarmComponent):
    """FLOWFarm farm AEP (Wh/yr); yaw_turbines in degrees.

    Options (``modeling_options["flowfarm"]``; defaults in parentheses):

    - ``jacobian_method`` ("reverse"): "reverse" (per-state Enzyme reverse; needs Julia 1.11+),
      "chunked" (auto: dense chunked ForwardDiff below ``auto_dense_below`` turbines, else a
      union-of-layouts sparse pattern self-checked against dense; alias "chunked_auto"),
      "chunked_dense", "chunked_sparse" (opt-in; cutoffs fixed at the start layout) or "di"
      (FLOWFarm's DifferentiationInterface sparse path).
    - ``opt_yaw`` (True): False drops the yaw partials; ``yaw_turbines`` must then stay 0.
    - ``auto_dense_below`` (25): turbine count below which "chunked" uses the dense path.
    - ``union_snapshots`` (8): layouts used to build the union sparsity cutoffs ("chunked" above
      the threshold; 0 disables the union).
    - ``union_seed`` (0): RNG seed of those layouts.
    - ``min_spacing_D`` (3.0): minimum turbine spacing (rotor diameters) of those layouts.
    - ``tolerance`` (1e-16): FLOWFarm sparsity tolerance.
    - wake model keys (deficit, deflection, turbulence, superposition, ...): see
      ``flowfarm_model.resolve_wake_model_inputs_for_flowfarm``.

    Partials of AEP_farm wrt x_turbines, y_turbines (m) and, if opt_yaw, yaw_turbines (deg).
    """

    def initialize(self):
        templates.FarmAEPTemplate.initialize(self)
        FLOWFarmComponent.initialize(self)

    def setup(self):
        templates.FarmAEPTemplate.setup(self)
        FLOWFarmComponent.setup(self)

    def setup_partials(self):
        self.declare_partials("AEP_farm", "x_turbines", method="exact")
        self.declare_partials("AEP_farm", "y_turbines", method="exact")
        if self.opt_yaw:
            self.declare_partials("AEP_farm", "yaw_turbines", method="exact")
            # yaw_turbines is in degrees; use a 0.01° step (not position-scale meters)
            self.set_check_partial_options("yaw_turbines", method="fd", step=0.01)

    def compute(self, inputs, outputs):
        FLOWFarmComponent._compute_aep(self, inputs, outputs)

    def compute_partials(self, inputs, partials):
        FLOWFarmComponent._compute_aep_partials(self, inputs, partials)


class FLOWFarmBatchPower(templates.BatchFarmPowerTemplate, FLOWFarmComponent):
    """FLOWFarm per-state farm power (W) and AEP_farm (Wh/yr); yaw_turbines in degrees.

    Options (``modeling_options["flowfarm"]``; defaults in parentheses):

    - ``jacobian_method`` ("reverse"): "reverse" (per-state Enzyme reverse; needs Julia 1.11+),
      "chunked" (auto: dense chunked ForwardDiff below ``auto_dense_below`` turbines, else a
      union-of-layouts sparse pattern self-checked against dense; alias "chunked_auto"),
      "chunked_dense", "chunked_sparse" (opt-in; cutoffs fixed at the start layout) or "di"
      (FLOWFarm's DifferentiationInterface sparse path).
    - ``opt_yaw`` (True): False drops the yaw partials; ``yaw_turbines`` must then stay 0.
    - ``auto_dense_below`` (25): turbine count below which "chunked" uses the dense path.
    - ``union_snapshots`` (8): layouts used to build the union sparsity cutoffs ("chunked" above
      the threshold; 0 disables the union).
    - ``union_seed`` (0): RNG seed of those layouts.
    - ``min_spacing_D`` (3.0): minimum turbine spacing (rotor diameters) of those layouts.
    - ``tolerance`` (1e-16): FLOWFarm sparsity tolerance.
    - wake model keys (deficit, deflection, turbulence, superposition, ...): see
      ``flowfarm_model.resolve_wake_model_inputs_for_flowfarm``.

    Partials of power_farm and AEP_farm wrt x_turbines, y_turbines (m) and, if opt_yaw, yaw_turbines (deg).
    """

    def initialize(self):
        templates.BatchFarmPowerTemplate.initialize(self)
        FLOWFarmComponent.initialize(self)

    def setup(self):
        templates.BatchFarmPowerTemplate.setup(self)
        FLOWFarmComponent.setup(self)

    def setup_partials(self):
        # State power sensitivities are provided via sparse_struct.state_gradients.
        self.declare_partials("power_farm", "x_turbines", method="exact")
        self.declare_partials("power_farm", "y_turbines", method="exact")
        # AEP_farm = sum(state_powers) is also an output; undeclared, its partials
        # would be treated as zero (silently zero d AEP_farm / d layout).
        self.declare_partials("AEP_farm", "x_turbines", method="exact")
        self.declare_partials("AEP_farm", "y_turbines", method="exact")
        if self.opt_yaw:
            self.declare_partials("power_farm", "yaw_turbines", method="exact")
            self.declare_partials("AEP_farm", "yaw_turbines", method="exact")
            # yaw_turbines is in degrees; use a 0.01° step (not position-scale meters)
            self.set_check_partial_options("yaw_turbines", method="fd", step=0.01)
        # x/y_turbines are in meters, spanning O(1e3). A large default FD step
        # (e.g. 1-10 m) can push a downwind turbine's effective wind speed
        # across a power-curve kink (cut-in, rated speed, cut-out), producing
        # a secant across the discontinuity instead of the true local
        # derivative -- this looks like a large partials error but is really
        # FD truncation noise unrelated to the analytic method's accuracy.
        # 0.01 m keeps the step well within a single power-curve segment for
        # essentially all wind states while staying far above float64
        # cancellation noise for O(1e4-1e6) W power values.
        self.set_check_partial_options("x_turbines", method="fd", step=0.01)
        self.set_check_partial_options("y_turbines", method="fd", step=0.01)

    def _get_state_scale(self):
        """Per-state scale factor used by FLOWFarm: hours_per_year * prob[i].

        FLOWFarm stores state_powers[i] = P_farm[i] * hours_per_year * prob[i]
        (AEP contribution in Wh/year, not instantaneous power in W). Divide by
        this factor to recover physical power in W for timeseries dispatch.
        """
        hours_per_year = 365.25 * 24.0
        wind_probs = np.asarray(
            self.farm.constants.wind_resource.wind_probabilities
        ).ravel()
        return hours_per_year * wind_probs  # shape (n_states,)

    def _primal_turbine_powers(self, x_eval_np):
        """Per-state turbine powers in W, shape (n_states, N); no derivatives."""
        global _PRIMAL_HELPER_DEFINED
        jl = self._jl
        if not _PRIMAL_HELPER_DEFINED:
            jl.seval(_PRIMAL_HELPER)
            _PRIMAL_HELPER_DEFINED = True
        self.farm.update_function(self.farm, to_julia_vector_float64(jl, x_eval_np))
        return np.asarray(jl.ard_state_turbine_powers(self.flowfarm_module, self.farm))

    def compute(self, inputs, outputs):
        # Forward pass only; the sparse gradient runs in compute_partials. Reuse it if
        # it was already evaluated at this x.
        x_eval_np = self._build_design_vector(inputs)
        scale = self._get_state_scale()
        if hasattr(self, "_cached_sparse_x") and np.array_equal(
            self._cached_sparse_x, x_eval_np
        ):
            state_powers_scaled = np.asarray(self.sparse_struct.state_powers).ravel()
            turbine_powers = np.asarray(self.sparse_struct.turbine_powers) / scale[np.newaxis, :]
        else:
            turbine_powers = self._primal_turbine_powers(x_eval_np).T  # (N, n_states), W
            state_powers_scaled = turbine_powers.sum(axis=0) * scale

        outputs["power_farm"] = state_powers_scaled / scale
        outputs["AEP_farm"] = float(state_powers_scaled.sum())  # Wh/year
        if (
            self.options["modeling_options"]
            .get("aero", {})
            .get("return_turbine_output")
        ):
            outputs["power_turbines"] = turbine_powers
            outputs["thrust_turbines"] = np.zeros(
                (self.N_turbines, self.N_wind_conditions)
            )

    def compute_partials(self, inputs, partials):
        x_eval_np = self._build_design_vector(inputs)
        self._evaluate_sparse(x_eval_np)

        # state_gradients[i, j] = d(state_powers[i])/d(x[j]) with same scale as state_powers.
        # Divide by scale to get d(power_farm[i])/d(x[j]) in W/m.
        scale = self._get_state_scale()  # shape (n_states,)
        state_gradients = np.asarray(self.sparse_struct.state_gradients) / scale[:, np.newaxis]
        partials["power_farm", "x_turbines"] = state_gradients[:, : self.N_turbines]
        partials["power_farm", "y_turbines"] = state_gradients[
            :, self.N_turbines : 2 * self.N_turbines
        ]
        if self.opt_yaw:
            # Julia gradient is d/d(yaw_rad); yaw_turbines input is in degrees → chain rule factor
            partials["power_farm", "yaw_turbines"] = (
                state_gradients[:, 2 * self.N_turbines : 3 * self.N_turbines] * np.deg2rad(1.0)
            )
        # AEP_farm = sum_i state_powers[i] (Wh/year), so its gradient is the column
        # sum of the unscaled state_gradients.
        aep_grad = np.asarray(self.sparse_struct.state_gradients).sum(axis=0)
        partials["AEP_farm", "x_turbines"] = aep_grad[: self.N_turbines]
        partials["AEP_farm", "y_turbines"] = aep_grad[self.N_turbines : 2 * self.N_turbines]
        if self.opt_yaw:
            partials["AEP_farm", "yaw_turbines"] = (
                aep_grad[2 * self.N_turbines : 3 * self.N_turbines] * np.deg2rad(1.0)
            )
