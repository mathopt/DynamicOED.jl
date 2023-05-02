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