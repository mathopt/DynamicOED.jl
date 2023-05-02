function modelingtoolkitize(prob::DAEProblem; kwargs...)


    prob.f isa DiffEqBase.AbstractParameterizedFunction && return prob.f.sys
    @parameters t

    p = prob.p

    has_p = !(p isa Union{DiffEqBase.NullParameters, Nothing})
    _vars = ModelingToolkit.define_vars(prob.u0, t)
    vars = prob.u0 isa Number ? _vars : ArrayInterface.restructure(prob.u0, _vars)
    params = if has_p
        _params = ModelingToolkit.define_params(p)
        p isa Number ? _params[1] :
        (p isa Tuple || p isa NamedTuple || p isa AbstractDict ? _params :
        ArrayInterface.restructure(p, _params))
    else
        []
    end

    D = Differential(t)
    Dvars = D.(vars)

    lhs = zero(prob.u0)
    if DiffEqBase.isinplace(prob)
        rhs = ArrayInterface.restructure(prob.u0, similar(vars, Num))
        fill!(rhs, 0)
        if prob.f isa ODEFunction &&
            prob.f.f isa FunctionWrappersWrappers.FunctionWrappersWrapper
            prob.f.f.fw[1].obj[](rhs, Dvars, vars, params, t)
        else
            prob.f(rhs, Dvars, vars, params, t)
        end
    else
        rhs = prob.f(Dvars, vars, params, t)
    end

    eqs = vcat([lhs[i] ~ rhs[i] for i in eachindex(prob.u0)]...)
    sts = vec(collect(vars))

    _params = params
    params = values(params)
    params = if params isa Number || (params isa Array && ndims(params) == 0)
        [params[1]]
    else
        vec(collect(params))
    end
    default_u0 = Dict(sts .=> vec(collect(prob.u0)))
    default_p = if has_p
        if prob.p isa AbstractDict
            Dict(v => prob.p[k] for (k, v) in pairs(_params))
        else
            Dict(params .=> vec(collect(prob.p)))
        end
    else
        Dict()
    end

    return ODESystem(eqs, t, sts, params,
                defaults = merge(default_u0, default_p);
                name = gensym(:MTKizedODE),
                tspan = prob.tspan,
                kwargs...)
end


function build_oed_dae_system(sys::ODESystem; tspan = ModelingToolkit.get_tspan(sys), ps = nothing,
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

    # Add new variables
    t = ModelingToolkit.get_iv(sys)
    D = Differential(t)

    eqs = map(x-> x.rhs - x.lhs, equations(sys))
    xs = states(sys)
    dxs = D.(xs)
    ps = isnothing(ps) ? [p for p in parameters(sys) if istunable(p) && !isinput(p)] : ps

    np, nx = length(ps), length(xs)
    fx = ModelingToolkit.jacobian(eqs, xs)
    fxs = ModelingToolkit.jacobian(eqs, dxs)
    fp = ModelingToolkit.jacobian(eqs, ps)
    hx = ModelingToolkit.jacobian(observed_rhs, xs)

    @variables (z(t))[1:length(observed_rhs)]=zeros(length(observed_rhs)) [description="Measurement State"]
    @parameters w[1:length(observed_rhs)]=ones(length(observed_rhs)) [description="Measurement function", tunable=true]
    @variables (F(t))[1:Int(np*(np+1)/2)]=zeros(Float64, (np,np)) [description="Fisher Information Matrix"]
    @variables (G(t))[1:nx, 1:np]=zeros(Float64, (nx,np)) [description="Sensitivity State"]
    @variables Q(t)[1:length(observed_rhs), 1:np] [description="Unweighted Fisher Information Derivative"]

    # Build the new system of deqs
    w = collect(w)
    G = collect(G)
    Q = collect(Q)
    dG = collect(D.(G))
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
            vec(0 .~ eqs);
            vec(0 .~ fxs*dG .+ fx*G  .+ fp);
            vec(D.(F) .~ collect(df));
            D.(z) .~ w
        ], tspan = tspan, observed = observed_eqs
    )
    return oed_system, F, G, z, observed_lhs, Q, observed_eqs, w
end