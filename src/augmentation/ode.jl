struct OEDFisher{PROB, G, DGDU, DGDP}
    problem::PROB
    g::G
    dgdu::DGDU
    dgdp::DGDP
end


function (fisher::OEDFisher)(u = fisher.problem.u0, p =fisher.problem.p; kwargs...)
    sol = __solve_fisher(fisher, u, p)
    __integrate_fisher(fisher, sol)

end

function _symmetric_from_vector(x::AbstractArray{T}, n::Int) where T
    return Symmetric([ i<=j ? x[Int(j*(j-1)/2+i)] : zero(T) for i=1:n, j=1:n])
end

function _symmetric_from_vector(x::AbstractArray{T}) where T
    # Find number n such that n*(n+1)/2 = length(x)
    @info x
    n = Int(sqrt(2 * length(x) + 0.25) - 0.5)
    _symmetric_from_vector(x, n)
end


function __integrate_fisher(fisher::OEDFisher, sol)
    np = size(sol.prob.p, 1)
    prob = IntegralProblem((t, p) -> fisher.g(sol(t), sol.prob.p, t), first(sol.t), last(sol.t); nout=np*(np+1)/2)
    fvec = solve(prob, HCubatureJL())
    _symmetric_from_vector(fvec.u)
end


function __solve_fisher(fisher::OEDFisher, u, p)
    _prob = remake(fisher.problem, u0 = u, p = p)
    _sol = solve(_prob, Tsit5())
end

function __∂fisher(fisher::OEDFisher, sol)
    du0, dp = adjoint_sensitivities(sol, Tsit5(), dgdu_continuous = fisher.dgdu, g = fisher.g, dgdp_continuous = fisher.dgdp, abstol = 1e-8, reltol = 1e-8) 
end   


function ChainRulesCore.rrule(fisher::OEDFisher, u0, p = fisher.problem.p)
    # Forwardpass
    sol = __solve_fisher(fisher, u0, p)
    F = __integrate_fisher(fisher, sol)

    function fisher_pullback(dg) # TODO What to do here
        du0, dp =__∂fisher(fisher, sol)
        return (NoTangent(), du0, dp)
    end

    F, fisher_pullback
end

function build_extended_dynamics(prob::ODEProblem)
    f = prob.f

    observed_f = (f.observed != SciMLBase.DEFAULT_OBSERVED) ? f.observed : (u,p,t) -> u.^2 .- p[1]

    nx = length(prob.u0)
    np = length(prob.p)

    x = make_variables(:u, nx)
    p = make_variables(:p, np)
    t = make_variables(:t, 1)

    eq = f(x, p, t)
    G = make_variables(:G, nx,np)
    dfdx = FastDifferentiation.jacobian(eq, x)

    dfdp = FastDifferentiation.jacobian(eq, p)

    Ġ = dfdx * G .+ dfdp
    Σ = vcat(eq, vec(Ġ))

    states = vcat(x, vec(G))
    state_syms = (isa(f, ODEFunction) && !isnothing(f.syms)) ? vcat(f.syms... , Symbol.(vec(G))) : Symbol.(states)
    parameters = p

    h = observed_f(states[1:nx], parameters, t)
    W = make_variables(:w, length(h))
    hx = FastDifferentiation.jacobian(h, states[1:nx])
    fidxs = tril!(trues((np,np)))

    fvec = sum(map(enumerate(W)) do (i, wi)
        hxiG = hx[i:i,:] * G
        wi * ((hxiG'hxiG)[fidxs])
    end)

    @info fvec



    jac_full = make_function(FastDifferentiation.jacobian(Σ, states), states, parameters, t; in_place = false)
    eq_full = make_function(Σ, states, parameters, t; in_place = false)
    f_full = make_function(fvec, states, parameters, t, W; in_place = false)

    dfdu = make_function(FastDifferentiation.jacobian(fvec, states), states, parameters, t, W; in_place = false)
    dfdp = make_function(FastDifferentiation.jacobian(fvec, parameters), states, parameters, t, W; in_place = false)

    @info FastDifferentiation.jacobian(fvec, states)
    @info FastDifferentiation.jacobian(fvec, parameters)


    # In place does not work as expected here, so we simply wrap
    f_new = let eq_full = eq_full
        (u, p, t) -> begin
            eq_full(vcat(u, p, t))
        end
    end
    
    
    jac_new = let jac_full = jac_full
        (u, p, t) -> begin
            jac_full(vcat(u, p, t))
        end
    end

    f_new_observed = let f_full = f_full, w_ = ones(length(h))
        (u, p, t) -> begin
            f_full(vcat(u, p, t, w_))
        end
    end

    dfdu_new = let dfdu = dfdu, w_ = ones(length(h))
        (u, p, t) -> begin 
            dfdu(vcat(u, p, t, w_))
        end
    end
    
    dfdp_new = let dfdp = dfdp, w_ = ones(length(h))
        (u, p, t) -> dfdp(vcat(u, p, t, w_))
    end
    


    new_f = ODEFunction{false}(f_new, jac = jac_new,
        syms = state_syms
    )

    new_prob = ODEProblem(new_f, vcat(prob.u0, zeros(length(G))), prob.tspan, prob.p)


    OEDFisher(new_prob, f_new_observed, dfdu_new, dfdp_new)

end
