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

function ExperimentalDesign(prob::ODEProblem, n::Int, kwargs...)
    Δt = -(reverse(prob.tspan)...)/n
    return ExperimentalDesign(prob, Δt, kwargs...)
end

function ExperimentalDesign(prob::ODEProblem, Δt, kwargs...)
    tgrid = get_tgrid(Δt, prob.tspan)
    sys = modelingtoolkitize(prob)
    oed_sys, F, G, z, Q, observed = build_oed_system(sys; tspan = first(tgrid), ps=parameters(sys), kwargs...)
    varmap = Dict(states(sys) .=> prob.u0)
    parmap, p0_old = get_parmap(parameters(sys), prob.p)
    oed_prob = ODEProblem(oed_sys, varmap, prob.tspan, parmap)
    ps_old = parameters(sys)
    ps_new = parameters(oed_sys)
    w_idx = length(ps_old)+1:length(ps_new)
    w = BitVector(Tuple(i ∈ w_idx ? true : false for i in 1:length(ps_new)))
    p0_new = Symbolics.getdefaultval.(ps_new[w_idx])
    p0 = eltype(p0_new).([p0_old; p0_new])
    return ExperimentalDesign{typeof(oed_sys), typeof(oed_prob), typeof(tgrid), typeof(p0), typeof(observed)}(oed_sys, oed_prob, w, tgrid, p0,
        (; F = F, G = G, z = z, Q = Q), sys, observed
    )
end

function get_parmap(ps::AbstractArray, p::NamedTuple)
    parmap, ps_ = Dict(), []
    parnames = keys(p)
    symbols  = Symbol.(ps)
    for name in parnames
        idx = argmax(symbols .== name)
        push!(ps_, p[name])
        push!(parmap, ps[idx] => p[name])
    end

    return parmap, ps_
end

function get_parmap(ps::AbstractArray, p::Tuple)
    parmap, ps_ = Dict(), []
    for (i, par) in enumerate(ps)
        push!(ps_, p[i])
        push!(parmap, par => p[i])
    end
    return parmap, ps_
end

function get_parmap(ps::AbstractArray, p::Dict)
    return get_parmap(ps, Tuple(values(p)))
end

function get_parmap(ps::AbstractArray, p::AbstractArray)
    return p, p
end

function ExperimentalDesign(sys::ODESystem, time_grid = [ModelingToolkit.get_tspan(sys)]; kwargs...)
    oed_sys, F, G, z, Q, observed = build_oed_system(sys; tspan = first(time_grid), kwargs...)
    oed_prob = ODEProblem(oed_sys)
    ps_old = parameters(sys)
    ps_new = parameters(oed_sys)
    w_idx = length(ps_old)+1:length(ps_new)
    w = BitVector(Tuple(i ∈ w_idx ? true : false for i in 1:length(ps_new)))
    p0 = Symbolics.getdefaultval.(ps_new)
    return ExperimentalDesign{typeof(oed_sys), typeof(oed_prob), typeof(time_grid), typeof(p0), typeof(observed)}(oed_sys, oed_prob, w, time_grid, p0,
        (; F = F, G = G, z = z, Q = Q), sys, observed
    )
end

function ExperimentalDesign(sys::ODESystem, n::Int; tspan = ModelingToolkit.get_tspan(sys), kwargs...)
    Δt = -(reverse(tspan)...)/n
    ExperimentalDesign(sys, Δt; tspan = tspan, kwargs...)
end

function ExperimentalDesign(sys::ODESystem, Δt::Real; tspan = ModelingToolkit.get_tspan(sys), kwargs...)
    tgrid = get_tgrid(Δt, tspan)
    ExperimentalDesign(
        sys, tgrid; kwargs...
    )
end

function get_tgrid(Δt::Real, tspan::Tuple{Real, Real})
    first_ts = first(tspan):Δt:(last(tspan)-Δt)
    last_ts = (first(tspan)+Δt):Δt:last(tspan)
    collect(zip(first_ts, last_ts))
end

Base.show(io::IO, oed::ExperimentalDesign) = show(io, oed.sys)
Base.summary(io::IO, oed::ExperimentalDesign) = summary(io, oed.sys)
Base.print(io::IO, oed::ExperimentalDesign) = print(io, oed.sys)


function build_oed_system(sys::ODESystem; tspan = ModelingToolkit.get_tspan(sys), ps=nothing, kwargs...)
    ## Get the eqs and the corresponding gradients
    simplified_sys = structural_simplify(sys)

    observed = ModelingToolkit.get_observed(sys)
    t = ModelingToolkit.get_iv(sys)
    if isempty(observed)
        observed_rhs = states(simplified_sys)
        @variables y(t)[1:length(observed_rhs)] [description = "Observed states"]
        y = collect(y)
        observed_lhs = y
    else
        observed_rhs = map(x->x.rhs, observed)
        observed_lhs = map(x->x.lhs, observed)
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
    @variables (F(t))[1:Int(np*(np+1)/2)]=zeros(Float64, (np,np)) [description="Fisher Information Matrix"] # Symmetric -> avoid redundancies!
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
    return structural_simplify(oed_system), F, G, z, Q, observed
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


function extract_sensitivities(oed::ExperimentalDesign, sol::AbstractArray)
    G_ = oed.variables.G
    G = vcat(map(enumerate(sol)) do (i,d)
        avoid_overlap = i == length(sol) ? 0 : 1
        d[G_][1:end-avoid_overlap]
    end...)

    hcat([g[:] for g in G]...)
end

function compute_local_information_gain(oed::ExperimentalDesign, x::NamedTuple, kwargs...)
    w, τ = x.w, x.τ
    Q = oed.variables.Q
    n_vars = size(Q,1)
    sol = oed(w; kwargs...)
    t = [d.t[i] for d in sol for i=1:length(d)-1]
    t = [t; last(ModelingToolkit.get_tspan(oed.sys_original))]
    Qs = reduce(vcat, map(enumerate(sol)) do (i,s)
        i < length(sol) ? s(s.t)[Q][1:end-1] : s(s.t)[Q]
    end)

    Pi = map(1:size(w, 1)) do i
        reduce(vcat, map(enumerate(Qs)) do (j,hG)
            hG[i,:]'*hG[i,:] / n_vars
        end)
    end
    return Pi, t, sol
end

function compute_global_information_gain(oed::ExperimentalDesign, x::NamedTuple, kwargs...)
    w, τ = x.w, x.τ
    F = oed.variables.F
    P, t, sol = compute_local_information_gain(oed, x, kwargs...)
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