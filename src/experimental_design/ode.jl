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

    oed_sys, F, G, z, h, hxG, observed, w = build_ode_oed_system(sys; tspan = first(tgrid), ps=ps, observed=observed_eqs, kwargs...)
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
    return ExperimentalDesign{typeof(oed_sys), typeof(oed_prob), typeof(tgrid), typeof(p0), typeof(prob), typeof(observed)}(oed_sys, oed_prob, w, tgrid, p0,
        (; F = F, G = G, z = z, h = h, hxG = hxG), sys, prob, observed
    )
end

"""
$(SIGNATURES)

Constructs the system of ordinary differential equations for the optimal experimental design
problem from a `ModelingToolkit.AbstractODESystem`.

Especially, the variables and differential equations for sensitivities and the Fisher
information matrix (FIM) are added to the system.
"""
function build_ode_oed_system(sys::ODESystem; tspan = ModelingToolkit.get_tspan(sys), ps = nothing,
    observed = nothing, simplify_system = false, kwargs...)
    ## Get the eqs and the corresponding gradients
    #sys = simplify_system ? structural_simplify(sys) : sys
    _observed = ModelingToolkit.get_observed(sys)
    t = ModelingToolkit.get_iv(sys)

    if isempty(_observed)
        if isnothing(observed)
            observed_rhs = states(sys)
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

    # Add new variables
    D = Differential(t)

    eqs = map(x-> x.rhs, equations(sys))
    xs = states(sys)
    ps = isnothing(ps) ? [p for p in parameters(sys) if istunable(p) && !isinput(p)] : ps

    np, nx = length(ps), length(xs)
    fx = ModelingToolkit.jacobian(eqs, xs)
    fp = ModelingToolkit.jacobian(eqs, ps)
    hx = ModelingToolkit.jacobian(observed_rhs, xs)

    @variables (z(t))[1:length(observed_rhs)]=zeros(length(observed_rhs)) [description="Measurement State"]
    @parameters w[1:length(observed_rhs)]=ones(length(observed_rhs)) [description="Measurement function", tunable=true]
    @variables (F(t))[1:Int(np*(np+1)/2)]=zeros(Float64, (np,np)) [description="Fisher Information Matrix"]
    @variables (G(t))[1:nx, 1:np]=zeros(Float64, (nx,np)) [description="Sensitivity State"]
    @variables Q(t)[1:length(observed_rhs), 1:np] [description="Unweighted Fisher Information Derivative"]

    # Build the new system of deqs
    w = collect(w)
    F = collect(F)
    G = collect(G)
    Q = collect(Q)
    hx = collect(hx)
    idxs = triu(ones(np,np)) .== 1.0
    df = sum(enumerate(w)) do (i, wi)
        wi*((hx[i:i, :]*G)'*(hx[i:i,:]*G))[idxs]
    end

    dynamic_eqs = equations(sys)

    sens_eqs =  Equation[(D.(G) .~ fx*G  .+ fp);]

    FIM_eqs = Equation[D.(F) .~ df;]

    observed_eqs = Equation[
        observed_lhs .~ observed_rhs;
        vec(Q .~  hx*G)
    ]

    @named oed_system = ODESystem([
            vec(dynamic_eqs);
            vec(sens_eqs);
            vec(FIM_eqs);
            D.(z) .~ w
        ], tspan = tspan, observed = observed_eqs
    )
    oed_system = simplify_system ? structural_simplify(oed_system) : oed_system
    return oed_system, F, G, z, observed_lhs, Q, observed_eqs, w
end