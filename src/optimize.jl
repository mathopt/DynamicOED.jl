"""
$(TYPEDEF)

A container for the solution of an OED problem.

# Fields
$(FIELDS)
"""
struct OEDSolution{C,P,Q,R,S} <: AbstractOEDSolution
    "Criterion"
    criterion::C
    "The solution of the system of ODEs."
    sol::P
    "Optimal sampling solution"
    w::Q
    "Information gain matrices and sensitivities"
    information_gain::R
    "Lagrange multipliers corresponding to sampling constraint"
    multiplier::S
    "Experimental design"
    oed::AbstractExperimentalDesign
end

function OEDSolution(oed, criterion, w; μ=nothing, kwargs...)
    n_vars = sum(oed.w_indicator)

    variables   =  (w = reshape(w[1:end-1], n_vars, :), regularization = w[end])

    P, t, sol   = compute_local_information_gain(oed, variables);
    Π, _, _     = compute_global_information_gain(oed, variables);
    G           = extract_sensitivities(oed, sol)

    information_gain = (t=t, local_information_gain = P, global_information_gain=Π,
                        sensitivities = G)

    return OEDSolution{typeof(criterion), typeof(sol), typeof(variables), typeof(information_gain), typeof(μ)}(
        criterion, sol, variables, information_gain, μ, oed
    )
end

# TODO: HOW TO DISPATCH ON SOLVERS? THEY ARE NOT KNOWN HERE!
function get_lagrange_multiplier(res)
    try
        return res.problem.mult_g
    catch e
        return nothing
    end
end

function SciMLBase.solve(ed::ExperimentalDesign, M::Float64, criterion::AbstractInformationCriterion, solver, options; integer = false, ad_backend = AD.ForwardDiffBackend(), kwargs...)
    # Define the loss and constraints

    n_exp = length(ed.tgrid)
    n_vars = sum(ed.w_indicator)

    tspan = ModelingToolkit.get_tspan(ed.sys_original)
    Δt = (last(tspan)-first(tspan))/n_exp
    n_measure = maximum([1, Int(floor(M/Δt))])

    loss(w) = inv(n_vars) * criterion(ed, reshape(w[1:end-1], n_vars, n_exp), w[end]; kwargs...) + w[end]

    m_constraints(w) = begin
        sol = last(ed(reshape(w[1:end-1], n_vars, n_exp); kwargs...))
        sol[end-n_vars+1:end,end] .- M
    end


    w_init = zeros(Float64, n_vars*n_exp)
    idxs = rand(1:n_vars*n_exp, n_measure)
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

    return OEDSolution(ed, criterion, res.minimizer, μ=multiplier, kwargs...)
end


function switching_function(res::OEDSolution{FisherACriterion})
    np  = sum(res.oed.w_indicator)
    sw = [tr.(P)/np for P in res.information_gain.local_information_gain]
    return (sw, "trace P(t)")
end

function switching_function(res::OEDSolution{ACriterion})
    np  = sum(res.oed.w_indicator)
    sw = [tr.(C)/np for C in res.information_gain.global_information_gain]
    return (sw, "trace Π(t)")
end

function switching_function(res::OEDSolution{DCriterion})
    F_ = res.oed.variables.F
    F = last(last(res.sol)[F_])
    detC = det(inv(F))
    sw = map(res.information_gain.global_information_gain) do Π
        detC .* [sum(F .* Πᵢ) for Πᵢ in Π]
    end
    return (sw,  "det C(tf) ⋅ ∑ F(tf) ∘ Π(t)")
end

function switching_function(res::OEDSolution{ECriterion})
    F_ = res.oed.variables.F
    F = last(last(res.sol)[F_])
    eigenC = eigen(inv(F))
    λ_max, idx_max = findmax(sign.(eigenC.values) .* abs.(eigenC.values))
    v = eigenC.vectors[:,idx_max:idx_max]
    sw = map(enumerate(res.information_gain.global_information_gain)) do (i,Π)
        [v' * Πᵢ * v for Πᵢ in Π]
    end
    return (sw, "v^T Π(t) v")
end

function _supported_criteria()
    return [FisherACriterion(), ACriterion(), DCriterion(), ECriterion()]
end
