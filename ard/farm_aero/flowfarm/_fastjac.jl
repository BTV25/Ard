# Chunked, color-seeded ForwardDiff Jacobian for FLOWFarm's unstable-pattern sparse path.
# Replaces the DifferentiationInterface jacobian! call; the primal evaluation, sparsity
# pattern detection and recoloring triggers are still FLOWFarm's own.
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

end # module
