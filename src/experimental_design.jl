"""
$(TYPEDEF)

A setup for an experimental design.

# Fields
$(FIELDS)
"""
struct ExperimentalDesign{S, P, T, PS, O} <: AbstractExperimentalDesign
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
    "Observed variables"
    observed::O
end

function ExperimentalDesign(prob::Union{<:ODEProblem, <:DAEProblem}, n::Int; params=nothing, observed=nothing, kwargs...)
    Δt = float(-(reverse(prob.tspan)...)/n)
    return ExperimentalDesign(prob, Δt, params=params, observed=observed; kwargs...)
end

"""
$(SIGNATURES)

Constructs an `ExperimentalDesign` from a given `ODEProblem`

Supported types for the parameters used in the `ODEProblem` are currently:
    - `NamedTuple`
    - `Dict`
    - `Tuple`
    - `AbstractArray`

The optional parameter `params` can be used to identify a subset of tunable parameters
of the system. Depending on the type of the `ODEProblem`'s parameters, it can be:
    - a `NamedTuple`,
    - an array containing a subset of keys of the `Dict`,
    - or an array of indices of the tunable parameters for `Tuple` and `AbstractArray`.

"""
function ExperimentalDesign(prob::ODEProblem, Δt::AbstractFloat; params=nothing, observed=nothing, kwargs...)
    tgrid = get_tgrid(Δt, prob.tspan)
    sys = ModelingToolkit.modelingtoolkitize(prob)
    sub_params = begin
        if !isnothing(params)
            params
        elseif typeof(prob.p) <: NamedTuple
            prob.p
        elseif typeof(prob.p) <: Dict
            collect(keys(prob.p))
        elseif typeof(prob.p) <: Tuple || typeof(prob.p) <: AbstractArray
            collect(1:length(prob.p))
        end
    end

    parmap, p0_old, ps = get_parmap(parameters(sys), prob.p, sub_params)

    # Define observed in the states of the modelingtoolkitized syste
    t_ = ModelingToolkit.get_iv(sys)
    p_ = parameters(sys)
    s_ = states(sys)
    observed_eqs, obs_ = begin
        if !isnothing(observed)
            obs_ = isnothing(observed) ? nothing : observed(prob.u0, prob.p, first(prob.tspan))
            @variables y(t_)[1:length(obs_)]
            y = collect(y)
            observed_eqs = Equation(y, observed(s_, p_, t_))
            observed_eqs, obs_
        else
            nothing, nothing
        end
    end

    oed_sys, F, G, z, h, hxG, observed, w = build_oed_system(sys; tspan = first(tgrid), ps=ps, observed=observed_eqs, kwargs...)
    varmap = Dict(states(sys) .=> prob.u0)

    param_permutation = find_permutation(parameters(sys), setdiff(parameters(oed_sys), w))
    p0_old = param_permutation * p0_old

    oed_prob = ODEProblem(oed_sys, varmap, prob.tspan, parmap)
    ps_old = parameters(sys)
    ps_new = parameters(oed_sys)
    w_idx = length(ps_old)+1:length(ps_new)
    w = BitVector(Tuple(i ∈ w_idx ? true : false for i in 1:length(ps_new)))
    p0_new = Symbolics.getdefaultval.(ps_new[w_idx])
    p0 = eltype(p0_new).([p0_old; p0_new])
    return ExperimentalDesign{typeof(oed_sys), typeof(oed_prob), typeof(tgrid), typeof(p0), typeof(observed)}(oed_sys, oed_prob, w, tgrid, p0,
        (; F = F, G = G, z = z, h = h, hxG = hxG), sys, observed
    )
end

function find_permutation(original_parameters, new_parameters)
    np = length(original_parameters)
    permmatrix = zeros(Int, np,np)
    for (i,diffp) in enumerate(new_parameters)
        for (j,orgp) in enumerate(original_parameters)
            @info diffp orgp isequal(diffp, orgp)
            if isequal(diffp, orgp)
                permmatrix[i,j] = 1
            end
        end
    end
    return permmatrix
end

function ExperimentalDesign(prob::DAEProblem, Δt::AbstractFloat; params=nothing, observed=nothing, kwargs...)
    tgrid = get_tgrid(Δt, prob.tspan)
    sys = modelingtoolkitize(prob)
    sub_params = begin
        if !isnothing(params)
            params
        elseif typeof(prob.p) <: NamedTuple
            prob.p
        elseif typeof(prob.p) <: Dict
            collect(keys(prob.p))
        elseif typeof(prob.p) <: Tuple || typeof(prob.p) <: AbstractArray
            collect(1:length(prob.p))
        end
    end

    parmap, p0_old, ps = get_parmap(parameters(sys), prob.p, sub_params)

    @info parmap, p0_old, ps
    # Define observed in the states of the modelingtoolkitized system
    t_ = ModelingToolkit.get_iv(sys)
    p_ = parameters(sys)
    s_ = states(sys)
    observed_eqs, obs_ = begin
        if !isnothing(observed)
            obs_ = isnothing(observed) ? nothing : observed(prob.u0, prob.p, first(prob.tspan))
            @variables y(t_)[1:length(obs_)]
            y = collect(y)
            observed_eqs = Equation(y, observed(s_, p_, t_))
            observed_eqs, obs_
        else
            nothing, nothing
        end
    end

    oed_sys, F, G, z, h, hxG, observed, w = build_oed_dae_system(sys; tspan = first(tgrid), ps=ps, observed=observed_eqs, kwargs...)

    D = Differential(ModelingToolkit.get_iv(oed_sys))

    param_permutation = find_permutation(parameters(sys), setdiff(parameters(oed_sys), w))
    p0_old = param_permutation * p0_old

    varmap = Dict(states(sys) .=> prob.u0)
    dvarmap = Dict(D.(states(sys)) .=> prob.du0)
    added_dvarmap = Dict(D.(setdiff(states(oed_sys), states(sys))) .=> 0.0)
    dvarmap = merge(dvarmap, added_dvarmap)
    @info parmap parameters(oed_sys)
    oed_prob = DAEProblem(oed_sys, dvarmap, varmap, prob.tspan, parmap)
    ps_old = parameters(sys)
    ps_new = parameters(oed_sys)
    w_idx = length(ps_old)+1:length(ps_new)
    w = BitVector(Tuple(i ∈ w_idx ? true : false for i in 1:length(ps_new)))
    p0_new = Symbolics.getdefaultval.(ps_new[w_idx])
    p0 = eltype(p0_new).([p0_old; p0_new])
    return ExperimentalDesign{typeof(oed_sys), typeof(oed_prob), typeof(tgrid), typeof(p0), typeof(observed)}(oed_sys, oed_prob, w, tgrid, p0,
        (; F = F, G = G, z = z, h = h, hxG = hxG), sys, observed
    )
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

Constructs an `ExperimentalDesign` from a given `ModelingToolkit.ODESystem`.

The parameter `time_grid` is a vector of tuples of timepoints, representing the intervals for
discretization of the problem.
"""
function ExperimentalDesign(sys::ODESystem, time_grid = [ModelingToolkit.get_tspan(sys)]; kwargs...)
    oed_sys, F, G, z, h, hxG, observed, w = build_oed_system(sys; tspan = first(time_grid), kwargs...)
    oed_prob = ODEProblem(oed_sys)
    ps_old = parameters(sys)
    ps_new = parameters(oed_sys)
    w_idx = length(ps_old)+1:length(ps_new)
    w = BitVector(Tuple(i ∈ w_idx ? true : false for i in 1:length(ps_new)))
    p0 = Symbolics.getdefaultval.(ps_new)
    return ExperimentalDesign{typeof(oed_sys), typeof(oed_prob), typeof(time_grid), typeof(p0), typeof(observed)}(oed_sys, oed_prob, w, time_grid, p0,
        (; F = F, G = G, z = z, h = h, hxG = hxG), sys, observed
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

"""
$(SIGNATURES)

Constructs the system of ordinary differential equations for the optimal experimental design
problem from a `ModelingToolkit.AbstractODESystem`.

Especially, the variables and differential equations for sensitivities and the Fisher
information matrix (FIM) are added to the system.
"""
function build_oed_system(sys::ODESystem; tspan = ModelingToolkit.get_tspan(sys), ps = nothing,
    observed = nothing, kwargs...)
    ## Get the eqs and the corresponding gradients
    simplified_sys = structural_simplify(sys)

    _observed = ModelingToolkit.get_observed(sys)
    t = ModelingToolkit.get_iv(sys)

    if isempty(_observed)
        if isnothing(observed)
            observed_rhs = states(simplified_sys)
            @variables y(t)[1:length(observed_rhs)] [description = "Observed states"]
            y = collect(y)
            observed_lhs = y
        else
            observed_rhs = observed.rhs
            observed_lhs = observed.lhs
        end
    else
        observed_rhs = map(x->x.rhs, _observed)
        observed_lhs = map(x->x.lhs, _observed)
    end

    eqs = map(x->x.rhs, equations(simplified_sys))
    xs = states(simplified_sys)
    ps = isnothing(ps) ? [p for p in parameters(simplified_sys) if istunable(p) && !isinput(p)] : ps
    np, nx = length(ps), length(xs)
    fx = ModelingToolkit.jacobian(eqs, xs)
    fp = ModelingToolkit.jacobian(eqs, ps)
    hx = ModelingToolkit.jacobian(observed_rhs, xs)

    # Add new variables
    t = ModelingToolkit.get_iv(simplified_sys)
    D = Differential(t)
    @variables (z(t))[1:length(observed_rhs)]=zeros(length(observed_rhs)) [description="Measurement State"]
    @parameters w[1:length(observed_rhs)]=ones(length(observed_rhs)) [description="Measurement function", tunable=true]
    @variables (F(t))[1:Int(np*(np+1)/2)]=zeros(Float64, (np,np)) [description="Fisher Information Matrix"]
    @variables (G(t))[1:nx, 1:np]=zeros(Float64, (nx,np)) [description="Sensitivity State"]
    @variables Q(t)[1:length(observed_rhs), 1:np] [description="Unweighted Fisher Information Derivative"]

    # Build the new system of deqs
    w = collect(w)
    G = collect(G)
    Q = collect(Q)
    hx = collect(hx)
    idxs = triu(ones(np,np)) .== 1.0
    df = sum(enumerate(w)) do (i, wi)
        wi*((hx[i:i, :]*G)'*(hx[i:i,:]*G))[idxs]
    end

    observed_eqs = Equation[
        observed_lhs .~ observed_rhs;
        vec(Q .~  hx*G)
    ]

    @named oed_system = ODESystem([
            equations(sys);
            vec(D.(G) .~ fx*G .+ fp);
            vec(D.(F) .~ collect(df));
            D.(z) .~ w
        ], tspan = tspan, observed = observed_eqs
    )
    return structural_simplify(oed_system), F, G, z, observed_lhs, Q, observed_eqs, w
end

# General predict
function (oed::ExperimentalDesign)(;u0::AbstractVector = oed.prob.u0, tspan::Tuple = ModelingToolkit.get_tspan(oed.sys), w::AbstractVector = oed.ps[oed.w_indicator], alg = Tsit5(), kwargs...)
    ps = vcat(oed.ps[.! oed.w_indicator], w)
    _prob = remake(oed.prob, u0 = u0, tspan = tspan, p = ps)
    solve(_prob, alg, kwargs...)
end

# Predict using the measurement array
function (oed::ExperimentalDesign)(w::AbstractArray; alg = Tsit5(), kwargs...)
    sol_ = nothing
    map(1:size(w, 2)) do i
        ps = vcat(oed.ps[.! oed.w_indicator], w[:,i])
        _prob = remake(oed.prob, u0 = i >= 2 ? sol_[:, end] : oed.prob.u0 , tspan = oed.tgrid[i], p = ps)
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
    sol = oed(w; kwargs...)
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
    F_inv = det(F_) > 1e-05 ? inv(F_) : nothing
    while isnothing(F_inv)
        F_ += 1e-6*I
        if det(F_) > 1e-02
            F_inv = inv(F_)
        end
    end
    Πi = isnothing(F_inv) ? nothing : map(1:size(w, 1)) do i
        Pi = P[i]
        map(Pi) do P_i
            F_inv*P_i*F_inv
        end
    end

    return Πi, t, sol
end