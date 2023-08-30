struct OEDFisher{PROB, O} <: AbstractFisher
    problem::PROB
    observed::O
    nh::Int
end

function (fisher::OEDFisher)(w, u = fisher.problem.u0, p =fisher.problem.p, tspan=fisher.problem.tspan;
                             kwargs...)
    sol = __solve_fisher(fisher, u, p, tspan)
    __integrate_fisher(fisher, sol, w)
end

function __integrate_fisher(fisher::OEDFisher, sol, w)
    p       = sol.prob.p
    np      = size(sol.prob.p, 1)
    nout    = Int(np*(np+1)/2)
    prob = IntegralProblem((t, p) -> fisher.observed(sol(t), sol.prob.p, t, p), first(sol.t), last(sol.t), w; nout=nout)
    F = Integrals.solve(prob, HCubatureJL())
    return F
    #u       = Array(sol)
    #solt    = sol.t
    #difft   = Zygote.@ignore diff(sol.t)
    #dF      = [fisher.observed(u[:,i],p,solt[i],w) for i in eachindex(solt)]
    #sum([0.5 * Δt * (dF[i] .+ dF[i+1]) for (i,Δt) in enumerate(difft)])
end


function __solve_fisher(fisher::OEDFisher, u, p, tspan; kwargs...)
    _prob = remake(fisher.problem, u0 = u, p = p, tspan=tspan)
    OrdinaryDiffEq.solve(_prob, Tsit5(), kwargs...)
end

function build_extended_dynamics(prob::ODEProblem)
    f = prob.f

    observed_f = (f.observed != SciMLBase.DEFAULT_OBSERVED) ? f.observed : (u,p,t) -> u

    nx = length(prob.u0)
    np = length(prob.p)

    x = make_variables(:u, nx)
    p = make_variables(:p, np)
    t = make_variables(:t, 1)
    @info p x t

    eq = f(x, p, t)
    G = make_variables(:G, nx,np)

    @info "Set up variables..."
    dfdx = FastDifferentiation.jacobian(eq, x)
    @info "Set up dfdx..."

    dfdp = FastDifferentiation.jacobian(eq, p)
    @info "Set up dfdp..."

    Ġ = dfdx * G .+ dfdp
    Σ = vcat(eq, vec(Ġ))

    states = vcat(x, vec(G))
    state_syms = (isa(f, ODEFunction) && !isnothing(f.syms)) ? vcat(f.syms... , Symbol.(vec(G))) : Symbol.(states)
    parameters = p

    h = observed_f(states[1:nx], parameters, t)
    nh = length(h)
    W = make_variables(:w, nh)
    hx = FastDifferentiation.jacobian(h, states[1:nx])
    fidxs = tril!(trues((np,np)))

    fvec = sum(map(enumerate(W)) do (i, wi)
        hxiG = hx[i:i,:] * G
        wi * ((hxiG'hxiG)[fidxs])
    end)
    @info "Set up dF..."


    #jac_full = make_function(FastDifferentiation.jacobian(Σ, states), states, parameters, t; in_place = false)
    eq_full = make_function(Σ, states, parameters, t; in_place = false)
    dF = make_function(fvec, states, parameters, t, W; in_place = false)

    @info "Set up eqfull and stuff...."
    # In place does not work as expected here, so we simply wrap
    f_new = let eq_full = eq_full
        (u, p, t) -> begin
            eq_full(vcat(u, p, t))
        end
    end


    #jac_new = let jac_full = jac_full
    #    (u, p, t) -> begin
    #        jac_full(vcat(u, p, t))
    #    end
    #end

    f_new_observed = let dF = dF
        (u, p, t, w_) -> begin
            dF(vcat(u, p, t, w_))
        end
    end

    new_f = ODEFunction{false}(f_new,# jac = jac_new,
        syms = state_syms
    )

    new_prob = ODEProblem(new_f, vcat(prob.u0, zeros(length(G))), prob.tspan, prob.p)


    OEDFisher(new_prob, f_new_observed, nh)
end


struct OEDProblem{T,S} <: AbstractExperimentalDesign
    predictor:: AbstractFisher
    timegrid::T
    Δt::Real
    sols::S
end



function OEDProblem(prob::ODEProblem, n::Int; kwargs...)
    aug_prob = DynamicOED.build_extended_dynamics(prob)
    Δt = float(-(reverse(prob.tspan)...)/n)
    timegrid = get_tgrid(Δt, prob.tspan)

    u0 = aug_prob.problem.u0
    p = aug_prob.problem.p
    sols = map(timegrid) do tgrid
        sol = __solve_fisher(aug_prob, u0, p, tgrid; kwargs...)
        u0 = last(sol)
        sol
    end

    return OEDProblem(aug_prob, timegrid, Δt, sols)
end



function (x::OEDProblem)(w::AbstractArray; kwargs...)

    F_ = sum(map(enumerate(zip(x.sols, eachcol(w)))) do (i,d)
        sol, w_i = d
        __integrate_fisher(x.predictor, sol, w_i)
    end)
    _symmetric_from_vector(F_)
end

function (x::OEDProblem)(w::AbstractArray, u0::AbstractArray; kwargs...)
    u_ = u0
    p = x.predictor.problem.p
    sols = map(x.timegrid) do tgrid
        sol = __solve_fisher(x.predictor, u_, p, tgrid; kwargs...)
        u_ = last(sol)
        sol
    end
    F_ = sum(map(enumerate(zip(sols, eachcol(w)))) do (i,d)
        sol, w_i = d
        __integrate_fisher(x.predictor, sol, w_i)
    end)
    _symmetric_from_vector(F_)
end

function SciMLBase.solve(x::OEDProblem, M::Union{<:Real, AbstractVector{<:Real}},
                 criterion::AbstractInformationCriterion; solver=IpoptAlg(),
                 options=IpoptOptions(), ad_backend=AbstractDifferentiation.ForwardDiffBackend(),
                 kwargs...)

    ngrid   = length(x.timegrid)
    nh      = x.predictor.nh

    if isa(M, AbstractVector)
        @assert length(M) == nh "Number of measurement constraints must be equal to the number of observed variables or scalar!"
    else
        M = M*ones(typeof(M), nh)
    end

    loss(w::AbstractArray) = apply_criterion(criterion, x, w; kwargs...)/nh

    m_constraints(w) = let Δt = x.Δt
        map(_x->Δt*sum(_x), eachrow(w)) .- M
    end

    w_init      = 0.1*ones(nh,ngrid)
    w_lower     = 0.0*ones(nh,ngrid)
    w_upper     = ones(nh,ngrid)

    loss = abstractdiffy(loss, ad_backend, w_init)
    m_constraints = abstractdiffy(m_constraints, ad_backend, w_init)

    model = Nonconvex.Model(loss)
    addvar!(model, w_lower, w_upper, init=w_init)#, integer = x_integer)
    #set_objective!(model, loss)
    add_ineq_constraint!(model, m_constraints)
    #setmin!(model, 1, (;w=zeros(n_vars, n_exp), τ = 0.0, iv=iv_lower))

    # Solve
    Nonconvex.optimize(model, solver, w_init, options = options)
end