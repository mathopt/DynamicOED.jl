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

    function OEDProblem(system::S, objective::O, )

    end
end

