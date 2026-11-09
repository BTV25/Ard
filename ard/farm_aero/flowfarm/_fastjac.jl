# Chunked ForwardDiff gradients of the farm power, two variants:
#  * DenseState / dense_gradient!: dense chunked ForwardDiff (every design variable seeded,
#    no sparsity pattern, no pruning; exact to roundoff). Default forward fallback.
#  * FastCache / gradient!: color-seeded chunked Jacobian on FLOWFarm's unstable-pattern sparse
#    path ("chunked_sparse"). Replaces the DifferentiationInterface jacobian! call; the primal
#    evaluation, sparsity pattern detection and recoloring triggers are still FLOWFarm's own.
module ArdFastJac

const FF = Main.FLOWFarm
const FD = FF.ForwardDiff

mutable struct FastCache
    cols::Vector{Any}                 # columns_by_color per state (nothing = replan)
    ws::Dict{Tuple{Int,Int},Any}      # (chunk width, prealloc id) => Dual workspace
end
FastCache(n_states::Int) = FastCache(Any[nothing for _ = 1:n_states], Dict{Tuple{Int,Int},Any}())

function workspace!(c::FastCache, w::Int, pid::Int, n_x::Int, n_t::Int)
    return get!(c.ws, (w, pid)) do
        D = FF.dual_type(FF._wind_state_power_ad!, Float64, w)
        (xd = zeros(D, n_x), pd = zeros(D, n_t))
    end
end

function color_columns(J, alg)
    colors = FF.column_colors(FF.coloring(J, FF.ColoringProblem(), alg))
    cols = [Int[] for _ = 1:maximum(colors)]
    for (j, k) in enumerate(colors)
        k > 0 && push!(cols[k], j)
    end
    return cols
end

@inline mk(::Type{FD.Dual{T,Float64,W}}, v::Float64, lane::Int) where {T,W} =
    FD.Dual{T,Float64,W}(v, FD.Partials(ntuple(i -> i == lane ? 1.0 : 0.0, Val(W))))

function chunk_jacobian!(xd::Vector{D}, pd, J, cols, x, farm, state, pid, lock) where {T,W,D<:FD.Dual{T,Float64,W}}
    fill!(J.nzval, 0.0)
    ncol = length(cols)
    for first = 1:W:ncol
        @inbounds for i in eachindex(x)
            xd[i] = mk(D, x[i], 0)
        end
        for lane = 1:W
            k = first + lane - 1
            k <= ncol || break
            for col in cols[k]
                xd[col] = mk(D, x[col], lane)
            end
        end
        FF._wind_state_power_ad!(pd, xd, farm, state, pid, lock)
        for lane = 1:W
            k = first + lane - 1
            k <= ncol || break
            for col in cols[k], p = J.colptr[col]:(J.colptr[col+1]-1)
                J.nzval[p] = FD.partials(pd[J.rowval[p]])[lane]
            end
        end
    end
    return nothing
end

function state_gradient!(ss, c::FastCache, x, state, pid, lock)
    res = ss.farm.constants.wind_resource
    if res.wind_speeds[state] == 0.0 || res.wind_probabilities[state] == 0.0
        return nothing
    end
    pow = view(ss.turbine_powers, :, state)
    FF.calculate_wind_state_power!(pow, x, ss.farm, state; prealloc_id=pid, lock=lock)
    FF.calculate_unstable_sparsity_pattern!(ss, x, state, pid)   # may call recolor (resets preps)
    J = ss.jacobians[state]
    if isnothing(ss.preps[state][])
        c.cols[state] = color_columns(J, ss.coloring_algorithm)
        ss.preps[state][] = true
    end
    w = ss.tier_widths[ss.current_tier[state]]
    wsp = workspace!(c, w, pid, length(x), size(ss.turbine_powers, 1))
    chunk_jacobian!(wsp.xd, wsp.pd, J, c.cols[state], x, ss.tier_farms[ss.current_tier[state]], state, pid, lock)
    ss.state_powers[state] = sum(pow)
    for j = 1:size(J, 2)
        s = 0.0
        for i = 1:size(J, 1)
            s += J[i, j]
        end
        ss.state_gradients[state, j] = s
    end
    return nothing
end

function gradient!(farm, x, ss, c::FastCache)
    nth = Threads.nthreads()
    ns = length(farm.constants.wind_resource.wind_probabilities)
    if nth > 1 && ns > 1
        per, rem = divrem(ns, nth)
        n = per + (rem > 0)
        assignments = 1:n:ns
        for ia in eachindex(assignments), w in ss.tier_widths
            workspace!(c, w, ia, length(x), size(ss.turbine_powers, 1))
        end
        l = Threads.SpinLock()
        Threads.@threads for ia in eachindex(assignments)
            i0 = assignments[ia]
            for i = i0:min(i0 + n - 1, ns)
                state_gradient!(ss, c, x, i, ia, l)
            end
        end
    else
        for i = 1:ns
            state_gradient!(ss, c, x, i, 1, nothing)
        end
    end
    farm.AEP[1] = sum(ss.turbine_powers)
    for j in axes(ss.state_gradients, 2)
        farm.AEP_gradient[j] = sum(@view ss.state_gradients[:, j])
    end
    return farm.AEP[1], farm.AEP_gradient
end

# ---- dense chunked ForwardDiff ------------------------------------------------------------

dual_type(w::Int) = FF.dual_type(FF._wind_state_power_ad!, Float64, w)

# Same field names the Python side reads from FLOWFarm's sparse struct. `farm` is a WindFarm
# whose preallocations are dual-typed with chunk width w (built in Python).
struct DenseState{D,F}
    farm::F
    w::Int
    ws::Vector{Any}   # per-thread (xd, pd) dual workspaces
    state_gradients::Matrix{Float64}
    state_powers::Vector{Float64}
    turbine_powers::Matrix{Float64}
end
function DenseState(farm, n_states::Int, n::Int, ndv::Int, w::Int)
    D = dual_type(w)
    ws = Any[(xd = zeros(D, ndv), pd = zeros(D, n)) for _ = 1:Threads.nthreads()]
    return DenseState{D,typeof(farm)}(farm, w, ws, zeros(n_states, ndv), zeros(n_states), zeros(n, n_states))
end

function dense_state!(ds::DenseState{D}, x, state, pid, lock) where {T,W,D<:FD.Dual{T,Float64,W}}
    res = ds.farm.constants.wind_resource
    ndv = length(x)
    n = size(ds.turbine_powers, 1)
    sg = view(ds.state_gradients, state, :)
    fill!(sg, 0.0)
    if res.wind_speeds[state] == 0.0 || res.wind_probabilities[state] == 0.0
        fill!(view(ds.turbine_powers, :, state), 0.0)
        ds.state_powers[state] = 0.0
        return nothing
    end
    xd = ds.ws[pid].xd
    pd = ds.ws[pid].pd
    for first = 1:W:ndv
        @inbounds for i = 1:ndv
            xd[i] = mk(D, x[i], 0)
        end
        for lane = 1:W
            col = first + lane - 1
            col <= ndv || break
            xd[col] = mk(D, x[col], lane)
        end
        FF._wind_state_power_ad!(pd, xd, ds.farm, state, pid, lock)
        if first == 1
            tot = 0.0
            for i = 1:n
                v = FD.value(pd[i])
                ds.turbine_powers[i, state] = v
                tot += v
            end
            ds.state_powers[state] = tot
        end
        for lane = 1:W
            col = first + lane - 1
            col <= ndv || break
            s = 0.0
            for i = 1:n
                s += FD.partials(pd[i])[lane]
            end
            sg[col] = s
        end
    end
    return nothing
end

function dense_gradient!(farm, x, ds::DenseState)
    nth = Threads.nthreads()
    ns = length(ds.state_powers)
    if nth > 1 && ns > 1
        per, rem = divrem(ns, nth)
        n = per + (rem > 0)
        assignments = 1:n:ns
        l = Threads.SpinLock()
        Threads.@threads for ia in eachindex(assignments)
            i0 = assignments[ia]
            for i = i0:min(i0 + n - 1, ns)
                dense_state!(ds, x, i, ia, l)
            end
        end
    else
        for i = 1:ns
            dense_state!(ds, x, i, 1, nothing)
        end
    end
    return sum(ds.state_powers), vec(sum(ds.state_gradients; dims = 1))
end

end # module
