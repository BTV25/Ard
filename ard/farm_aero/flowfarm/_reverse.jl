# Per-state Enzyme reverse gradient of the farm power (one row of state_gradients per state),
# threaded over contiguous state chunks, one workspace per chunk. Does not use FLOWFarm's
# sparse struct and never calls farm.update_function: x/y(/yaw) are passed directly.
module ArdReverse

using Enzyme
const FF = Main.FLOWFarm
const ENZ_REV = Enzyme.set_runtime_activity(Enzyme.Reverse)

# SpecialFunctions' gamma rule lowers to a bare C digamma Enzyme's JIT cannot find: supply it.
function Enzyme.EnzymeRules.augmented_primal(
    config::Enzyme.EnzymeRules.RevConfigWidth{1}, ::Enzyme.Const{typeof(FF.SpecialFunctions.gamma)},
    ::Type{<:Enzyme.Active}, x::Enzyme.Active{Float64})
    value = FF.SpecialFunctions.gamma(x.val)
    primal = Enzyme.EnzymeRules.needs_primal(config) ? value : nothing
    return Enzyme.EnzymeRules.AugmentedReturn(primal, nothing, value * FF.SpecialFunctions.digamma(x.val))
end
function Enzyme.EnzymeRules.reverse(
    ::Enzyme.EnzymeRules.RevConfigWidth{1}, ::Enzyme.Const{typeof(FF.SpecialFunctions.gamma)},
    dret::Enzyme.Active, derivative, ::Enzyme.Active{Float64})
    return (derivative * dret.val,)
end

struct Case{V,C,M}
    turbine_z::V
    hub_height::V
    turbine_yaw::V
    rotor_diameter::V
    constants::C
    model_set::M   # constants.model_set, or its no-deflection twin on the fixed-zero-yaw path
end

struct Scratch{V,M,I}
    rot_x::V; rot_y::V; velocities::V; ct::V; ai::V; local_ti::V
    wake_deficits::M; contribution_matrix::M; deflections::M; sigma_squared::M
    sort_index::I
end
Scratch(n::Int) = Scratch(ntuple(_ -> zeros(n), 6)..., ntuple(_ -> zeros(n, n), 4)..., zeros(Int, n))
function zero_scratch!(s::Scratch)
    for f in fieldnames(Scratch)
        fill!(getfield(s, f), 0)
    end
end

struct Workspace{S,V}
    scratch::S; shadow::S; power::V; dpower::V; dx::V
end
Workspace(n::Int, ndv::Int) = Workspace(Scratch(n), Scratch(n), zeros(n), zeros(n), zeros(ndv))

# opt_yaw = true: x = [x; y; yaw], all active. opt_yaw = false: x = [x; y] active, yaw Const (all zero).
function state_power!(power, x, yaw_fixed, scratch, case, state_id)
    n = length(power)
    resource = case.constants.wind_resource
    if FF.below_cutin_speed(resource, state_id, case.turbine_z, case.hub_height, case.constants.cut_in_speed)
        @inbounds for i in eachindex(power)
            power[i] = 0.0
        end
        return nothing
    end
    # yaw is one concrete type per method instance (a Union with case.turbine_yaw breaks Enzyme)
    yaw = yaw_fixed === nothing ? (@view x[2n+1:3n]) : yaw_fixed
    FF.rotate_to_wind_direction!(scratch.rot_x, scratch.rot_y, (@view x[1:n]), (@view x[n+1:2n]),
        resource.wind_directions[state_id])
    sortperm!(scratch.sort_index, scratch.rot_x; alg=Base.Sort.InsertionSort)
    # yaw is one concrete SubArray type (a Union with case.turbine_yaw breaks Enzyme type analysis)
    FF.turbine_velocities_one_direction_vel(scratch.rot_x, scratch.rot_y, case.turbine_z, case.rotor_diameter,
        case.hub_height, yaw, scratch.sort_index, case.constants.ct_models,
        case.constants.rotor_sample_points_y, case.constants.rotor_sample_points_z, resource,
        case.model_set; wind_farm_state_id=state_id, turbine_velocities=scratch.velocities,
        turbine_ct=scratch.ct, turbine_ai=scratch.ai, turbine_local_ti=scratch.local_ti,
        wake_deficits=scratch.wake_deficits, contribution_matrix=scratch.contribution_matrix,
        deflections=scratch.deflections, sigma_squared=scratch.sigma_squared)
    FF.turbine_powers_one_direction(case.constants.generator_efficiency, case.constants.cut_in_speed,
        case.constants.cut_out_speed, case.constants.rated_speed, case.constants.rated_power,
        case.rotor_diameter, scratch.velocities, yaw, resource.air_density,
        case.constants.power_models; wt_power=power)
    return nothing
end

function state_objective!(x, yaw_fixed, power, scratch, case, state_id)
    state_power!(power, x, yaw_fixed, scratch, case, state_id)
    obj = zero(eltype(x))
    @inbounds for p in power
        obj += p
    end
    return obj
end

function reverse_state!(ws::Workspace, x, yaw_fixed, case, state_id)
    fill!(ws.dx, 0.0); fill!(ws.dpower, 0.0); zero_scratch!(ws.shadow)
    Enzyme.autodiff(ENZ_REV, state_objective!, Enzyme.Active,
        Enzyme.Duplicated(x, ws.dx), Enzyme.Const(yaw_fixed), Enzyme.Duplicated(ws.power, ws.dpower),
        Enzyme.Duplicated(ws.scratch, ws.shadow), Enzyme.Const(case), Enzyme.Const(state_id))
    return nothing
end

# Same field names the Python side reads from FLOWFarm's sparse struct.
struct ReverseState{C,W,Y}
    case::C
    ws::Vector{W}
    xy::Vector{Float64}   # active [x; y] when opt_yaw = false (unused otherwise)
    yaw_fixed::Y          # nothing (yaw active) or the fixed all-zero yaw vector
    state_gradients::Matrix{Float64}
    state_powers::Vector{Float64}
    turbine_powers::Matrix{Float64}
    warm::Base.RefValue{Bool}
end

# opt_yaw = false (yaw not a design variable, must be 0): yaw is Const, the yaw columns of
# state_gradients stay 0, and GaussYawVariableSpreadDeflection (identically 0 at zero yaw, with
# zero x/y derivatives there) is replaced by NoYawDeflection. Primal unchanged; ~1.4x faster.
function ReverseState(farm, n_states::Int, n::Int, ndv::Int; opt_yaw::Bool=true)
    ms = farm.constants.model_set
    if !opt_yaw && ms.wake_deflection_model isa FF.GaussYawVariableSpreadDeflection
        ms = FF.WindFarmModelSet(ms.wake_deficit_model, FF.NoYawDeflection(), ms.wake_combination_model,
            ms.local_ti_model, ms.point_velocity_average_factor)
    end
    case = Case(farm.constants.turbine_z, farm.hub_height, farm.turbine_yaw, farm.rotor_diameter, farm.constants, ms)
    nact = opt_yaw ? ndv : 2n
    return ReverseState(case, [Workspace(n, nact) for _ = 1:Threads.nthreads()], zeros(2n),
        opt_yaw ? nothing : zeros(n), zeros(n_states, ndv), zeros(n_states), zeros(n, n_states), Ref(false))
end


function gradient!(farm, x::Vector{Float64}, rs::ReverseState)
    ns = length(rs.state_powers); nc = length(rs.ws)
    if rs.yaw_fixed === nothing
        xa = x
    else
        n = length(rs.yaw_fixed)
        all(iszero, @view x[2n+1:3n]) || error("FLOWFarm opt_yaw is false but yaw_turbines is nonzero; set opt_yaw: true.")
        rs.xy .= @view x[1:2n]
        xa = rs.xy
    end
    if !rs.warm[]   # compile every chunk's workspace serially before threading (Enzyme JIT is not thread-safe)
        for c = 1:nc
            reverse_state!(rs.ws[c], xa, rs.yaw_fixed, rs.case, 1)
        end
        rs.warm[] = true
    end
    # one workspace per task; states handed out by an atomic counter (state costs vary: cut-in skips, wake count)
    next = Threads.Atomic{Int}(1)
    Threads.@threads :static for c = 1:nc
        ws = rs.ws[c]
        while (s = Threads.atomic_add!(next, 1)) <= ns
            reverse_state!(ws, xa, rs.yaw_fixed, rs.case, s)
            f = 365.25 * 24.0 * rs.case.constants.wind_resource.wind_probabilities[s]   # FLOWFarm's AEP weighting
            @inbounds for j in eachindex(ws.dx)
                rs.state_gradients[s, j] = ws.dx[j] * f
            end
            tot = 0.0
            @inbounds for i in eachindex(ws.power)
                rs.turbine_powers[i, s] = ws.power[i] * f
                tot += ws.power[i] * f
            end
            rs.state_powers[s] = tot
        end
    end
    return sum(rs.state_powers), vec(sum(rs.state_gradients; dims=1))
end

end # module
