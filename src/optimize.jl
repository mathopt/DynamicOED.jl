"""
$(TYPEDEF)

The solution to an optimal experimental design problem for a certain criterion.
The sampling decisions can be retrieved from `w`.

# Fields
$(FIELDS)
"""
struct OEDSolution{C,P,Q,T,GI,S,M,O} <: AbstractOEDSolution
    "Criterion"
    criterion::C
    "The solution of the system of ODEs."
    sol::P
    "Optimal sampling solution"
    w::Q
    "Time grids"
    t::T
    "Information gain matrices"
    information_gain::GI
    "Sensitivities"
    sensitivities::S
    "Lagrange multipliers corresponding to sampling constraint"
    multiplier::M
    "Objective"
    obj::O
    "Experimental design"
    oed::AbstractExperimentalDesign
end

function OEDSolution(oed, criterion, w, obj; μ=nothing, kwargs...)
    P, t, sol   = compute_local_information_gain(oed, w; kwargs...);
    Π, _, _     = compute_global_information_gain(oed, w; local_information_gain=(P,t,sol), kwargs...);
    G           = extract_sensitivities(oed, sol)

    information_gain = (local_information_gain = P, global_information_gain=Π,)

    return OEDSolution{typeof(criterion), typeof(sol), typeof(w), typeof(t), typeof(information_gain), typeof(G), typeof(μ), typeof(obj)}(
        criterion, sol, w, t, information_gain, G, μ, obj, oed
    )
end

"""
$(SIGNATURES)

Solves an optimal experimental design problem which is defined through the `ExperimentalDesign`.

The upper bound on the measurements is given by `M` which may be a single upper bound for all
observed quantities or a vector of upper bounds.

Initial values for the sampling decisions can be passed through `w_init`, which is expected
to be of size `(n_vars,n_exp)` where `n_vars` is the number of measurement functions and `n_exp` is
the number of intervals from the discretization of the `ExperimentalDesign`.

If `integer` is set to `true`, the variables representing the sampling decisions
are treated as integer variables.
"""
function SciMLBase.solve(ed::ExperimentalDesign, M::Union{<:Real, AbstractVector{<:Real}},
    criterion::AbstractInformationCriterion, solver, options; integer = false, ad_backend = AD.ForwardDiffBackend(),
    w_init = nothing, kwargs...)

    # Define the loss and constraints
    n_exp = length(ed.tgrid)
    n_vars = sum(ed.w_indicator)
    z = ed.variables.z

    if isa(M, AbstractVector)
        @assert length(M) == n_vars "Number of measurement constraints must be equal to the number of observed variables or scalar!"
    else
        M = M*ones(typeof(M), n_vars)
    end

    tspan = ModelingToolkit.get_tspan(ed.sys_original)
    Δt = (last(tspan)-first(tspan))/n_exp
    n_measure = map(x -> maximum([1,x]),  Int.(floor.(M./Δt)))

    loss(x) = apply_criterion(criterion, ed, x; kwargs...)/n_vars + x.τ

    m_constraints(x) = begin
        sol = last(ed(reshape(x.w, n_vars, n_exp); kwargs...))
        sol[z][end] .- M
    end

    w_init = isnothing(w_init) ? begin
        y = zeros(Float64, n_vars,n_exp)
        for i=1:n_vars
            idxs = rand(1:n_exp, n_measure[i])
            y[:,idxs] .= one(Float64)
        end
        y
    end : w_init
    @assert length(w_init) == n_vars * n_exp "Provided initial value must have correct dimension. Got $(length(w_init)), expected $(n_vars*n_exp)."

    w_lower = 0.5*ones(Float64, n_vars, n_exp)
    w_upper = ones(Float64, n_vars, n_exp)

    τ_init = eltype(w_init)(1e-5)
    τ_lower = eltype(w_init)(1e-4)
    τ_upper = typeof(τ_init)(1)

    x_init = (; w = w_lower, τ = τ_init)
    x_upper = (; w = w_upper, τ = τ_upper)
    x_lower = (; w = w_lower, τ = τ_lower)
    x_integer = (; w = integer ? ones(Bool, size(w_init)) : zeros(Bool, size(w_init)), τ = false)

    ## Convert to  AD
    loss = abstractdiffy(loss, ad_backend, x_init)
    m_constraints = abstractdiffy(m_constraints, ad_backend, x_init)

    model = Nonconvex.Model(loss)
    addvar!(model, x_lower, x_upper, init=x_init, integer = x_integer)
    #set_objective!(model, loss)
    add_ineq_constraint!(model, m_constraints)
    setmin!(model, 1, (;w=zeros(n_vars, n_exp), τ = 0.0))

    # Solve
    res = Nonconvex.optimize(model, solver, x_init, options = options)

    multiplier = get_lagrange_multiplier(res)

    return OEDSolution(ed, criterion, res.minimizer, res.minimum; μ=multiplier, kwargs...)
end


function get_lagrange_multiplier(res)
    try
        return res.problem.mult_g
    catch e
        return nothing
    end
end