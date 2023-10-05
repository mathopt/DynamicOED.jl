function SciMLBase.solve(prob::OEDProblem, M::Union{<:Real, AbstractVector{<:Real}},
    criterion::AbstractInformationCriterion; w_init=.1, solver=IpoptAlg(), integer=false,
    options=IpoptOptions(), ad_backend=AbstractDifferentiation.ForwardDiffBackend(),
    bounds_u0 = nothing, kwargs...)

    nh = prob.predictor.dimensions.nh

    if isa(M, AbstractVector)
        @assert length(M) == nh "Number of measurement constraints must be equal to the number of observed variables or scalar!"
    else
        M = M*ones(typeof(M), nh)
    end

    loss(w::W) where W = apply_criterion(criterion, prob, w; kwargs...)/nh

    m_constraints(w) = let Δt = -(reverse(prob.predictor.problem.tspan)...)/length(prob.timegrid.grids.wgrid)
        map(_x->Δt*sum(_x), eachrow(_get_w(w))) .- M
    end

    x_init, x_lower, x_upper, x_integer = make_params(prob, w_init, integer, bounds_u0)

    loss = abstractdiffy(loss, ad_backend, x_init)
    m_constraints = abstractdiffy(m_constraints, ad_backend, x_init)

    model = Nonconvex.Model(loss)
    addvar!(model, x_lower, x_upper, init=x_init, integer = x_integer)
    add_ineq_constraint!(model, m_constraints)

    # Solve
    res = Nonconvex.optimize(model, solver, x_init, options = options)
    μ = get_lagrange_multiplier(res)
    OEDSolution(prob, criterion, res.minimizer, res.minimum; μ=μ, kwargs...);
end