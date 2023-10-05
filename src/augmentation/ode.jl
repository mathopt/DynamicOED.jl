struct OEDFisher{PROB, I, O} <: AbstractFisher
    problem::PROB
    integrand::I
    observed::O
    dimensions::NamedTuple
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
        (t, p) -> fisher.integrand(sol(t), p0, t, p)
    end
    _prob    = Integrals.IntegralProblem{false}(integrand, first(sol.t), last(sol.t), w; nout=nout)
    F = solve(_prob, HCubatureJL())
    return F
end

function build_extended_dynamics(prob::ODEProblem; parameters=1:length(prob.p), variable_iv=false)
    f = prob.f

    observed_f = (f.observed != SciMLBase.DEFAULT_OBSERVED) ? f.observed : (u,p,t) -> u

    FastDifferentiation.clear_cache()

    nx = length(prob.u0)
    np_original = length(prob.p)
    np_selected = length(parameters)
    np_iv = nx

    np_real = variable_iv ? np_selected + np_iv : np_selected
    np_created = variable_iv ? np_original + np_iv : np_original # just for dimension purposes

    x = make_variables(:u, nx)
    p = make_variables(:p, np_created)
    t = make_variables(:t, 1)

    eq = f(x, p, t)
    G = make_variables(:G, nx, np_real)
    G_iv =  zeros(nx, np_selected)
    G_iv = variable_iv ? hcat(G_iv, Matrix{eltype(G_iv)}(I,nx,nx)) : G_iv

    dfdx = FastDifferentiation.jacobian(eq, x)
    parameters_considered = variable_iv ? vcat(parameters, collect(np_original+1:np_created)) : parameters
    dfdp = FastDifferentiation.jacobian(eq, p[parameters_considered])
    G1 = dfdx * G .+ dfdp

    Σ = eltype(eq).(vcat(eq, vec(G1)))

    states = vcat(x, vec(G))
    state_syms = (isa(f, ODEFunction) && !isnothing(f.syms)) ? vcat(f.syms... , Symbol.(vec(G))) : Symbol.(states)

    h_ = observed_f(states[1:nx], p, t)
    h  = make_function(h_, states[1:nx], p, t)
    nh = length(h_)
    W = make_variables(:w, nh)
    hx = FastDifferentiation.jacobian(h_, states[1:nx])
    hx_fun = make_function(hx, states[1:nx], p, t)
    fidxs = triu!(trues((np_real,np_real)))

    fvec = sum(map(enumerate(W)) do (i, wi)
        hxiG = hx[i:i,:] * G
        wi * ((hxiG' * hxiG)[fidxs])
    end)

    #jac_full = make_function(FastDifferentiation.jacobian(Σ, states), states, p, t; in_place = false)
    eq_full = make_function(Σ, states, p, t; in_place = false)
    dF = make_function(fvec, states, p, t, W; in_place = false)

    f_new = let eq_full = eq_full
        (u, p, t) -> begin
            eq_full(vcat(u, p, t))
        end
    end

    integrand = let dF = dF
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

    dimensions = (nx=nx, nh=nh, np=np_real)
    observed = (h=h, hx=hx_fun,)

    OEDFisher(new_prob, integrand, observed, dimensions)
end

function __solve_fisher(fisher::OEDFisher, u, p, tspan; alg=Tsit5(), kwargs...)
    _prob = remake(fisher.problem, u0 = u, p = p, tspan=tspan)
    OrdinaryDiffEq.solve(_prob, alg, kwargs...)
end

struct OEDProblem{ufixed,S} <: AbstractExperimentalDesign
    predictor:: AbstractFisher
    timegrid::AbstractTimeGrid
    sols::S
end

function OEDProblem(predictor::AbstractFisher, timegrid::AbstractTimeGrid, Δt::Real, sols::S; variable_iv::Bool = false) where S
    OEDProblem{variable_iv, S}(predictor, timegrid, Δt, sols)
end

function OEDProblem(prob::DEProblem, nw::Int; nc=nw, parameters=1:length(prob.p), variable_iv = false, kwargs...)
    aug_prob = DynamicOED.build_extended_dynamics(prob; parameters=parameters, variable_iv =variable_iv)

    timegrid = TimeGrid(prob.tspan, nw, nc)

    u0 = aug_prob.problem.u0
    p = aug_prob.problem.p
    sols = grid_solve(aug_prob, u0, p, tuple(timegrid.simgrid...); kwargs...)

    return OEDProblem{variable_iv, typeof(sols)}(aug_prob, timegrid, sols)
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

function grid_integrate(problem::AbstractFisher, indicator::NamedTuple, w::AbstractArray,
                            sols::T; kwargs...) where T <: Tuple
    _grid_integrate(problem, indicator, w, sols...; kwargs...)
end

function _grid_integrate(problem::AbstractFisher, indicator::NamedTuple, w::AbstractArray,
                            single_sol::T, sols...; kwargs...) where T

    i, sol = single_sol
    wi = w[:,indicator.w[i]]
    F = __integrate_fisher(problem, sol, wi)
    F .+ _grid_integrate(problem, indicator, w, sols...)
end

function _grid_integrate(problem::AbstractFisher, indicator::NamedTuple, w::AbstractArray,
                            single_sol::T; kwargs...) where T

    i, sol = single_sol
    wi = w[:,indicator.w[i]]
    __integrate_fisher(problem, sol, wi)
end

function (x::OEDProblem)(w::AbstractArray; kwargs...)
    F_ = grid_integrate(x.predictor, x.timegrid.indicator, w, tuple(enumerate(x.sols)...))
    _symmetric_from_vector(F_)
end

function (x::OEDProblem)(w::AbstractArray, u0::AbstractVector; kwargs...)
    sols = grid_solve(x.predictor, u0, x.predictor.problem.p, tuple(x.timegrid.simgrid...); kwargs...)
    F_ = grid_integrate(x.predictor, x.timegrid.indicator, w, tuple(enumerate(sols)...))
    _symmetric_from_vector(F_)
end

function (x::OEDProblem)(w::AbstractArray, u0::AbstractVector, p::AbstractVector; kwargs...)
    sols = grid_solve(x.predictor, u0, p, tuple(x.timegrid.simgrid...); kwargs...)
    F_ = grid_integrate(x.predictor, x.timegrid.indicator, w,  tuple(enumerate(sols)...))
    _symmetric_from_vector(F_)
end

function (x::OEDProblem)(w::NamedTuple; kwargs...)
    u0_ = vcat(w.u0, x.predictor.problem.u0[size(w.u0,1)+1:end])
    x(w.w, u0_; kwargs...)
end

_get_w(w::AbstractArray) = w
_get_w(w::NamedTuple) = w.w

_make_w(prob::OEDProblem, init::Union{Real, AbstractArray{<:Real}}) = begin
    nw = length(prob.timegrid.grids.wgrid)
    nh = prob.predictor.dimensions.nh

    w_init = begin
        if isa(init, AbstractArray)
            @assert size(init) == (nh,nw) "Number of measurement constraints must be equal to the number of observed variables or scalar!"
            init
        else
            init*ones(typeof(init), nh,nw)
        end
    end
    w_init
end

_make_u0(prob::OEDProblem) = begin
    u0 = prob.predictor.problem.u0
    nx = prob.predictor.dimensions.nx
    u0[1:nx]
end

_make_c0(prob::OEDProblem) = begin

end

function make_params(prob::OEDProblem{false}, init::Union{Real, AbstractArray{<:Real}}, integer::Bool, args...)
    w = _make_w(prob, init)
    w_lower = zeros(eltype(w),size(w))
    w_upper = ones(eltype(w), size(w))
    w_integer = integer ? ones(Bool, size(w)) : zeros(Bool, size(w))
    return w, w_lower, w_upper, w_integer
end

function make_params(prob::OEDProblem{true}, init::Union{Real, AbstractArray{<:Real}}, integer::Bool, bounds)
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
    x_integer = (w = integer ? ones(Bool, size(w)) : zeros(Bool, size(w)), u0=zeros(Bool, size(u0)))

    return x, x_lower, x_upper, x_integer
end