"""
$(TYPEDEF)

A container for the solution of an OED problem.

# Fields
$(FIELDS)
"""
struct OEDSolution{P,Q,R,S} <: AbstractOEDSolution
    "The solution of the system of ODEs."
    sol::P
    "Optimal sampling solution"
    w::Q
    "Information gain matrices"
    information_gain::R
    "Lagrange multipliers corresponding to sampling constraint"
    multiplier::S
end

function OEDSolution(oed, w; μ=nothing, kwargs...)
    n_vars = sum(oed.w_indicator)

    variables =  (w_opt = reshape(w[1:end-1], n_vars, :), ϵ = w[end])

    P, t, sol = DynamicOED.compute_local_information_gain(oed, variables.w_opt);
    Π, _, _ = DynamicOED.compute_global_information_gain(oed, variables.w_opt);

    information_gain = (t=t, local_information_gain = P, global_information_gain=Π)

    return OEDSolution{typeof(sol), typeof(variables), typeof(information_gain), typeof(μ)}(
        sol, variables, information_gain, μ
    )
end

# TODO: HOW TO DISPATCH ON SOLVERS, THEY ARE NOT KNOWN HERE!
function get_lagrange_multiplier(res)
    try
        return res.problem.mult_g
    catch e
        return nothing
    end
end

function SciMLBase.solve(ed::ExperimentalDesign, M::Int, criterion::AbstractInformationCriterion, solver, options; integer = false, ad_backend = AD.ForwardDiffBackend(), kwargs...)
    # Define the loss and constraints

    n_exp = length(ed.tgrid)
    n_vars = sum(ed.w_indicator)

    loss(w) = inv(n_vars) * criterion(ed, reshape(w[1:end-1], n_vars, n_exp), w[end]; kwargs...) + w[end]

    m_constraints(w) = begin
        sol = last(ed(reshape(w[1:end-1], n_vars, n_exp); kwargs...))
        sol[end-n_vars+1:end,end] .- M
    end


    w_init = zeros(Float64, n_vars*n_exp)
    idxs = rand(1:n_vars*n_exp, M)
    w_init[idxs] .= one(Float64)
    w_init = [w_init; 1e-04]
    model = Nonconvex.Model(loss)
    # Temporary lower bound to avoid failure of code in initial constraint evaluation
    addvar!(model, 0.5*ones(Float64, n_vars*n_exp), ones(Float64, n_vars*n_exp), init=w_init, integer = integer ? ones(Bool, n_vars*n_exp) : zeros(Bool, n_vars*n_exp))
    addvar!(model, 0.5*one(Float64), one(Float64), init=1e-4)
    add_ineq_constraint!(model, m_constraints)
    ## Convert to the backend
    ad_model = abstractdiffy(model, ad_backend)
    setmin!(model, zeros(Float64, n_vars*n_exp+1))
    # Solve
    res = Nonconvex.optimize(ad_model, solver, w_init, options = options)

    multiplier = get_lagrange_multiplier(res)

    return OEDSolution(ed, res.minimizer, μ=multiplier, kwargs...)
end