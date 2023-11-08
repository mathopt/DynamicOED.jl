"""
$(TYPEDEF)

The basic definition of an optimal experimental design problem.

# Fields

$(FIELDS)
"""
struct OEDProblem{S, O, T, C, A, DO}
    "The optimal experimental design system in form of an ODESystem"
    system::S
    "The objective criterion"
    objective::O
    "The time grid"
    timegrid::T
    "Constraints"
    constraints::C 
    "Solver for the differential equations"
    alg::A
    "Differential equations options"
    diffeq_options::DO
end

function OEDProblem(sys::ModelingToolkit.AbstractODESystem, objective::AbstractInformationCriterion, constraints = []; alg = Tsit5(), tspan = ModelingToolkit.get_tspan(sys), diffeqoptions::NamedTuple = NamedTuple(), kwargs...)
    OEDProblem(sys, objective, Timegrid(sys, tspan), constraints, alg, diffeqoptions)
end

function build_predictor(prob::OEDProblem)
    tspan = (first(first(prob.timegrid.timespans)), last(last(prob.timegrid.timespans)))
    odae_prob = ODAEProblem(prob.system, Pair[], tspan)
    remaker = OEDRemake(prob.system, tspan, prob.timegrid)
    
    predictor = let remaker = remaker, odae_prob = odae_prob, alg = prob.alg, options = prob.diffeq_options
        (p) ->  sequential_solve(remaker, odae_prob, alg, p; options...)
    end
end



function Optimization.OptimizationProblem(prob::OEDProblem, AD::Optimization.ADTypes.AbstractADType; integer_constraints::Bool = false)
    tspan = (first(first(prob.timegrid.timespans)), last(last(prob.timegrid.timespans)))
    odae_prob = ODAEProblem(prob.system, Pair[], tspan)
    remaker = OEDRemake(prob.system, tspan, prob.timegrid)
    
    solver = let remaker = remaker, odae_prob = odae_prob, alg = prob.alg, options = prob.diffeq_options
        (p) ->  sequential_solve(remaker, odae_prob, alg, p; options...)
    end

    f_idxs = [is_fisher_state(xi) for xi in states(prob.system)]
    n = Val(Int(sqrt(2 * sum(f_idxs) + 0.25) - 0.5))

    objective = let solver = solver, criterion = prob.objective, idx = f_idxs, n = n
        (p, x) -> begin
            x, t = solver(p) 
            F = _symmetric_from_vector(x[idx, end], n)
            criterion(F) 
        end
    end



    p0 = Float64.(generate_initial_variables(prob.system, prob.timegrid))
    lb = Float64.(generate_variable_bounds(prob.system, prob.timegrid, true))
    ub = Float64.(generate_variable_bounds(prob.system, prob.timegrid, false))

    cons = let p_prototype = 0. * p0 
        (res, p, x) -> begin 
            p_ = p + p_prototype
            res[1] = sum(p_.controls)
            res[2] = sum(p_.measurements)
        end
    end

    integrality = Bool.(p0 * 0)
    if integer_constraints
        integrality.controls .= true
        integrality.measurements .= true
    end
    syms = Symbol.(vcat(get_initial_conditions(prob.system), get_control_parameters(prob.system), get_measurement_function(prob.system)))
    
    opt_f = OptimizationFunction(
        objective, AD; 
        syms = syms,
        cons = cons,
    )

    OptimizationProblem(opt_f, p0, lb = lb, ub = ub, int = integrality, 
        lcons = [1.0, 1.0], ucons = [2.0, 3.0]
    )
end