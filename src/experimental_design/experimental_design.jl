"""
$(TYPEDEF)

A setup for an experimental design.

# Fields
$(FIELDS)
"""
struct ExperimentalDesign{S, P, T, PS, PO, O} <: AbstractExperimentalDesign
    "The ODESystem"
    sys::S
    "The corresponding ODEProblem"
    prob::P
    "Indicator which parameters are used for measurements"
    w_indicator::BitVector
    "The time grid"
    tgrid::T
    "The (fixed) parameters"
    ps::PS
    "The named tuple of (extended) variables"
    variables::NamedTuple
    "The original system for a-posteriori calculation of information gain"
    sys_original::S
    "Original ODE/DAE problem"
    prob_original::PO
    "Observed variables"
    observed::O
end


function ExperimentalDesign(prob::Union{<:ODEProblem, <:DAEProblem}, n::Int; params=nothing, observed=nothing, kwargs...)
    Δt = float(-(reverse(prob.tspan)...)/n)
    return ExperimentalDesign(prob, Δt, params=params, observed=observed; kwargs...)
end


"""
$(SIGNATURES)

Constructs an `ExperimentalDesign` from a given `ModelingToolkit.ODESystem`.

The parameter `time_grid` is a vector of tuples of timepoints, representing the intervals for
discretization of the problem.
"""
function ExperimentalDesign(sys::ODESystem, time_grid = [ModelingToolkit.get_tspan(sys)]; kwargs...)
    oed_sys, F, G, z, h, hxG, observed, w = build_ode_oed_system(sys; tspan = first(time_grid), kwargs...)
    oed_prob = ODEProblem(oed_sys)
    ps_old = parameters(sys)
    ps_new = parameters(oed_sys)
    w_idx = length(ps_old)+1:length(ps_new)
    w = BitVector(Tuple(i ∈ w_idx ? true : false for i in 1:length(ps_new)))
    p0 = Symbolics.getdefaultval.(ps_new)
    return ExperimentalDesign{typeof(oed_sys), typeof(oed_prob), typeof(time_grid), typeof(p0), Nothing, typeof(observed)}(oed_sys, oed_prob, w, time_grid, p0,
        (; F = F, G = G, z = z, h = h, hxG = hxG), sys, nothing, observed
    )
end

function ExperimentalDesign(sys::ODESystem, n::Int; tspan = ModelingToolkit.get_tspan(sys), kwargs...)
    Δt = float(-(reverse(tspan)...)/n)
    ExperimentalDesign(sys, Δt; tspan = tspan, kwargs...)
end

function ExperimentalDesign(sys::ODESystem, Δt::AbstractFloat; tspan = ModelingToolkit.get_tspan(sys), kwargs...)
    tgrid = get_tgrid(Δt, tspan)
    ExperimentalDesign(
        sys, tgrid; kwargs...
    )
end




function find_permutation(original_parameters, new_parameters)
    np = length(original_parameters)
    permmatrix = zeros(Int, np,np)
    for (i,diffp) in enumerate(new_parameters)
        for (j,orgp) in enumerate(original_parameters)
            if isequal(diffp, orgp)
                permmatrix[i,j] = 1
            end
        end
    end
    return permmatrix
end



"""
$(SIGNATURES)

Transfer initial values for the parameters from the user-supplied `ODEProblem` to the newly
created `ODESystem`.
"""
function get_parmap(params_sys::AbstractArray, params_prob::NamedTuple, params::NamedTuple)
    parmap, p0, ps = Dict(), [], eltype(params_sys)[]
    parnames = keys(params_prob)
    symbols  = Symbol.(params_sys)
    for name in parnames
        idx = argmax(symbols .== name)
        push!(p0, params_prob[name])
        push!(parmap, params_sys[idx] => params_prob[name])
        if name in keys(params)
            push!(ps, params_sys[idx])
        end
    end
    return parmap, p0, ps
end

function get_parmap(params_sys::AbstractArray, params_prob::Tuple, idxs::AbstractArray{Int})
    parmap, p0, ps = Dict(), [], eltype(params_sys)[]
    for (i, par) in enumerate(params_sys)
        push!(p0, params_prob[i])
        push!(parmap, par => params_prob[i])
        if i in idxs
            push!(ps, par)
        end
    end
    return parmap, p0, ps
end

function get_parmap(params_sys::AbstractArray, params_prob::Dict, keys_ps::AbstractArray)
    idxs = [i for (i, parname) in enumerate(keys(params_prob)) if parname in keys_ps]
    return get_parmap(params_sys, Tuple(values(params_prob)), idxs)
end

function get_parmap(params_sys::AbstractArray, params_prob::AbstractArray, idxs::AbstractArray{Int})
    return params_prob, params_prob, params_sys[idxs]
end

"""
$(SIGNATURES)
Constructs a vector of time grids from a timespan and a stepsize.
"""
function get_tgrid(Δt::AbstractFloat, tspan::Tuple{Real, Real})
    @assert Δt < -(reverse(tspan)...) "Stepsize must be smaller than total time interval."
    first_ts = first(tspan):Δt:(last(tspan)-Δt)
    last_ts = (first(tspan)+Δt):Δt:last(tspan)
    tgrid = collect(zip(first_ts, last_ts))
    if !isapprox(last(last(tgrid)),last(tspan))
        push!(tgrid, (last(last(tgrid)), last(tspan)))
    end
    tgrid
end

Base.show(io::IO, oed::ExperimentalDesign) = show(io, oed.sys)
Base.summary(io::IO, oed::ExperimentalDesign) = summary(io, oed.sys)
Base.print(io::IO, oed::ExperimentalDesign) = print(io, oed.sys)

# General predict
function (oed::ExperimentalDesign)(;u0::AbstractVector = oed.prob.u0, tspan::Tuple = ModelingToolkit.get_tspan(oed.sys), w::AbstractVector = oed.ps[oed.w_indicator], alg = Tsit5(), kwargs...)
    ps = vcat(oed.ps[.! oed.w_indicator], w)
    _prob = remake(oed.prob, u0 = u0, tspan = tspan, p = ps)
    solve(_prob, alg, kwargs...)
end

# Predict using the measurement array
function (oed::ExperimentalDesign)(x::NamedTuple; alg = Tsit5(), kwargs...)
    sol_ = nothing
    sts = states(oed.sys)
    idxs_iv = [argmax(isequal.(sts_, sts)) for sts_ in states(oed.sys_original) if any(isequal.(sts_, sts))]
    not_found = [i for (i,sts_) in enumerate(states(oed.sys_original)) if !any(isequal.(sts_, sts))]
    found = setdiff(1:length(x.iv), not_found)

    first_initial_condition = eltype(x.iv).(oed.prob.u0)
    first_initial_condition[idxs_iv] = x.iv[found]

    map(1:size(x.w, 2)) do i
        ps = vcat(oed.ps[.! oed.w_indicator], x.w[:,i])
        _prob = remake(oed.prob, u0 = i >= 2 ? sol_[:, end] : first_initial_condition  , tspan = oed.tgrid[i], p = ps)
        sol_ = solve(_prob, alg, sensealg = ForwardDiffSensitivity(), kwargs...)
        return sol_
    end
end

"""
$(SIGNATURES)

Extracts sensitivities from solutions on all intervals.
"""
function extract_sensitivities(oed::ExperimentalDesign, sol::AbstractArray)
    G_ = oed.variables.G
    G = vcat(map(enumerate(sol)) do (i,d)
        avoid_overlap = i == length(sol) ? 0 : 1
        d[G_][1:end-avoid_overlap]
    end...)

    hcat([g[:] for g in G]...)
end

"""
$(SIGNATURES)

Returns the local information gain, the stacked vector of solution timesteps and the vector
containing an `ODESolution` for each discretization interval for iterate `x`.
"""
function compute_local_information_gain(oed::ExperimentalDesign, x::NamedTuple; kwargs...)
    w, τ = x.w, x.τ
    hxG = oed.variables.hxG
    n_vars = size(hxG,1)
    sol = oed(x; kwargs...)
    t = [d.t[i] for d in sol for i=1:length(d)-1]
    t = [t; last(ModelingToolkit.get_tspan(oed.sys_original))]
    Qs = reduce(vcat, map(enumerate(sol)) do (i,s)
        i < length(sol) ? s(s.t)[hxG][1:end-1] : s(s.t)[hxG]
    end)

    Pi = map(1:size(w, 1)) do i
        reduce(vcat, map(enumerate(Qs)) do (j,hG)
            hG[i,:]'*hG[i,:] / n_vars
        end)
    end
    return Pi, t, sol
end

"""
$(SIGNATURES)

Returns the global information gain, the stacked vector of solution timesteps and the vector
containing an `ODESolution` for each discretization interval for iterate `x`.
"""
function compute_global_information_gain(oed::ExperimentalDesign, x::NamedTuple; local_information_gain = nothing, kwargs...)
    w, τ = x.w, x.τ
    F = oed.variables.F
    P, t, sol = isnothing(local_information_gain) ? compute_local_information_gain(oed, x, kwargs...) : local_information_gain
    F_ = _symmetric_from_vector(last(last(sol)[F]))
    F_inv = inv(F_ + τ*I)

    Πi = isnothing(F_inv) ? nothing : map(1:size(w, 1)) do i
        Pi = P[i]
        map(Pi) do P_i
            F_inv*P_i*F_inv
        end
    end

    return Πi, t, sol
end