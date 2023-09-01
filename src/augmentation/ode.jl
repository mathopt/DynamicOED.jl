struct OEDFisher{PROB, O} <: AbstractFisher
    problem::PROB
    observed::O
    nxnh::NamedTuple
end

function (fisher::OEDFisher)(w, u = fisher.problem.u0, p =fisher.problem.p, tspan=fisher.problem.tspan;
                             kwargs...)
    sol = __solve_fisher(fisher, u, p, tspan)
    __integrate_fisher(fisher, sol, w)
end

function __integrate_fisher(fisher::OEDFisher, sol::DESolution, w::AbstractArray)
    p       = sol.prob.p
    np      = size(sol.prob.p, 1)
    nout    = Int(np*(np+1)/2)
    integrand = let p0 = p, sol = sol
        (t, p) -> fisher.observed(sol(t), p0, t, p)
    end
    _prob    = Integrals.IntegralProblem{false}(integrand, first(sol.t), last(sol.t), w; nout=nout)
    F = solve(_prob, HCubatureJL())
    return F
    #u       = Array(sol)
    #solt    = sol.t
    #difft   = Zygote.@ignore diff(sol.t)
    #dF      = [fisher.observed(u[:,i],p,solt[i],w) for i in eachindex(solt)]
    #sum([0.5 * Δt * (dF[i] .+ dF[i+1]) for (i,Δt) in enumerate(difft)])
end


function build_extended_dynamics(prob::ODEProblem; variable_iv=false)
    f = prob.f

    observed_f = (f.observed != SciMLBase.DEFAULT_OBSERVED) ? f.observed : (u,p,t) -> u

    FastDifferentiation.clear_cache()

    nx = length(prob.u0)
    np_or = length(prob.p)
    np_iv = nx

    np = variable_iv ? np_or + np_iv : np_or

    x = make_variables(:u, nx)
    p = make_variables(:p, np)
    t = make_variables(:t, 1)

    eq = f(x, p, t)
    G = make_variables(:G, nx,np)
    G_iv =  zeros(nx,np_or)
    G_iv = variable_iv ? hcat(G_iv, Matrix{eltype(G_iv)}(I,nx,nx)) : G_iv

    dfdx = FastDifferentiation.jacobian(eq, x)
    dfdp = FastDifferentiation.jacobian(eq, p)

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
        wi * ((hxiG' * hxiG)[fidxs])
    end)


    #jac_full = make_function(FastDifferentiation.jacobian(Σ, states), states, parameters, t; in_place = false)
    eq_full = make_function(Σ, states, parameters, t; in_place = false)
    dF = make_function(fvec, states, parameters, t, W; in_place = false)

    # In place does not work as expected here, so we simply wrap
    f_new = let eq_full = eq_full
        (u, p, t) -> begin
            eq_full(vcat(u, p, t))
        end
    end

    f_new_observed = let dF = dF
        (u, p, t, w_) -> begin
            dF(vcat(u, p, t, w_))
        end
    end

    new_f = ODEFunction{false}(f_new,# jac = jac_new,
        syms = state_syms
    )

    p =  variable_iv ? vcat(prob.p, prob.u0) : prob.p

    new_prob = ODEProblem(new_f, vcat(prob.u0, vec(G_iv)), prob.tspan, p)

    FastDifferentiation.clear_cache()

    nxnh = (nx=nx, nh=nh)

    OEDFisher(new_prob, f_new_observed, nxnh)
end

function __solve_fisher(fisher::OEDFisher, u, p, tspan; alg=Tsit5(), kwargs...)
    _prob = remake(fisher.problem, u0 = u, p = p, tspan=tspan)
    OrdinaryDiffEq.solve(_prob, alg, kwargs...)
end

struct OEDProblem{ufixed,T,S} <: AbstractExperimentalDesign
    predictor:: AbstractFisher
    timegrid::T
    Δt::Real
    sols::S
end

function OEDProblem(predictor::AbstractFisher, timegrid::T, Δt::Real, sols::S; variable_iv::Bool = false) where {T, S}
    OEDProblem{variable_iv, T, S}(predictor, timegrid, Δt, sols)
end


function OEDProblem(prob::DEProblem, n::Int; variable_iv = false, kwargs...)
    aug_prob = DynamicOED.build_extended_dynamics(prob; variable_iv =variable_iv)
    Δt = float(-(reverse(prob.tspan)...)/n)
    timegrid = get_tgrid(Δt, prob.tspan)

    u0 = aug_prob.problem.u0
    p = aug_prob.problem.p
    sols = grid_solve(aug_prob, u0, p, tuple(timegrid...); kwargs...)

    return OEDProblem(aug_prob, timegrid, Δt, sols; variable_iv)
end

function grid_solve(problem::AbstractFisher, u0::AbstractVector, p0::AbstractVector, tgrids::T; kwargs...) where T <: Tuple
    _grid_solve(problem, u0, p0, tgrids...; kwargs...)
end

function _grid_solve(problem::AbstractFisher, u0::AbstractVector, p0::AbstractVector, tgrid::T, grids...; kwargs...) where T
    sol = __solve_fisher(problem, u0, p0, tgrid; kwargs...)
    (sol, _grid_solve(problem, sol[:, end], p0, grids...; kwargs...)...)
end

function _grid_solve(problem::AbstractFisher, u0::AbstractVector, p0::AbstractVector, tgrid::T; kwargs...) where T
    sol = __solve_fisher(problem, u0, p0, tgrid; kwargs...)
    (sol,)
end

function grid_integrate(problem::AbstractFisher, x::T; kwargs...) where T <: Tuple
    _grid_integrate(problem, x...; kwargs...)
end

function _grid_integrate(problem::AbstractFisher, single_x::T, x...; kwargs...) where T
    F = __integrate_fisher(problem, single_x...)
    F .+ _grid_integrate(problem, x...)
end

function _grid_integrate(problem::AbstractFisher, single_x::T; kwargs...) where T
    __integrate_fisher(problem, single_x...)
end

function (x::OEDProblem)(w::AbstractArray; kwargs...)
    F_ = grid_integrate(x.predictor, tuple(zip(x.sols, eachcol(w))...))
    _symmetric_from_vector(F_)
end

function (x::OEDProblem)(w::AbstractArray, u0::AbstractVector; kwargs...)
    sols = grid_solve(x.predictor, u0, x.predictor.problem.p, tuple(x.timegrid...); kwargs...)
    F_ = grid_integrate(x.predictor, tuple(zip(sols, eachcol(w))...))
    _symmetric_from_vector(F_)
end

function (x::OEDProblem)(w::AbstractArray, u0::AbstractVector, p::AbstractVector; kwargs...)
    sols = grid_solve(x.predictor, u0, p, tuple(x.timegrid...); kwargs...)
    F_ = grid_integrate(x.predictor, tuple(zip(sols, eachcol(w))...))
    _symmetric_from_vector(F_)
end

_get_w(w::AbstractArray) = w
_get_w(w::NamedTuple) = w.w

_make_w(prob::OEDProblem, init::Union{Real, AbstractArray{<:Real}}) = begin
    ngrid   = length(prob.timegrid)
    nh      = prob.predictor.nxnh.nh

    w_init = begin
        if isa(init, AbstractArray)
            @assert size(init) == (nh,ngrid) "Number of measurement constraints must be equal to the number of observed variables or scalar!"
            init
        else
            init*ones(typeof(init), nh,ngrid)
        end
    end
    w_init
end

_make_u0(prob::OEDProblem) = begin
    u0 = prob.predictor.problem.u0
    nx = prob.predictor.nxnh.nx
    u0[1:nx]
end

_make_c0(prob::OEDProblem) = begin

end

make_params(prob::OEDProblem{false}, init::Union{Real, AbstractArray{<:Real}}, args...) = begin
    w = _make_w(prob, init)
    w_lower = zeros(eltype(w),size(w))
    w_upper = ones(eltype(w), size(w))

    return w, w_lower, w_upper
end

function make_params(prob::OEDProblem{true}, init::Union{Real, AbstractArray{<:Real}}, bounds)
    w = _make_w(prob, init)
    w_lower = zeros(size(w))
    w_upper = ones(size(w))

    u0 = _make_u0(prob)
    if isnothing(bounds)
        T = eltype(u0)
        bounds = (ntuple(i->T(-Inf), size(u0, 1)), ntuple(i->T(Inf), size(u0, 1)))
    end
    u0_lower = first(bounds)
    u0_upper = last(bounds)

    x = (w=w, u0=u0,)
    x_lower = (w = w_lower, u0=u0_lower,)
    x_upper = (w = w_upper, u0=u0_upper,)
    return x, x_lower, x_upper
end