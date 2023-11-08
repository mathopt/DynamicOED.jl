"""
$(TYPEDEF)

The basic definition of an optimal experimental design problem.

# Fields

$(FIELDS)
"""
struct OEDProblem{S, O, T, C}
    "The optimal experimental design system in form of an ODESystem"
    system::S
    "The objective criterion"
    objective::O
    "The time grid"
    timegrid::T
    "Constraints"
    constraints::C 
end

function generate_objective(sys::ModelingToolkit.AbstractODESystem, criterion::AbstractInformationCriterion)
    idx = [is_fisher_state(xi) for xi in states(sys) ]
    @assert any(idx) "No fisher information is provided! Did you process your system using `OEDSystem`?"
    

    n = Val(Int(sqrt(2*sum(idx) + 0.25) - 0.5)) # Precompute here

    function oed_objective(solution::SciMLBase.AbstractTimeseriesSolution{T}, τ::T = zero(T))::T where T
        f = solution[idx, end]
        F = _symmetric_from_vector(f, n)
        criterion(F, τ)
    end
end

generate_objective(problem::OEDProblem) = generate_objective(problem.system, problem.objective)

