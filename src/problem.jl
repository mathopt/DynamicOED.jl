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

function ModelingToolkit.states(prob::OEDProblem)
    grid = prob.timegrid
    vars = grid.variables
    grid_vars = []
    
    @inbounds for i in eachindex(vars)
        N = size(grid.timegrids[i], 1)
        push!(
            grid_vars, (
                vars[i], Symbolics.variables(vars[i], 1:N)
            )
        )
    end


    return sortkeys(NamedTuple(grid_vars)) 
end

function get_timegrids(prob::OEDProblem)
    grid = prob.timegrid
    vars = grid.variables
    grid_vars = []
    
    @inbounds for i in eachindex(vars)
        push!(
            grid_vars, (
                vars[i], copy(grid.timegrids[i])
            )
        )
    end


    return sortkeys(NamedTuple(grid_vars))
end

function build_predictor(prob::OEDProblem)
    tspan = (first(first(prob.timegrid.timespans)), last(last(prob.timegrid.timespans)))
    odae_prob = ODAEProblem(prob.system, Pair[], tspan)
    remaker = OEDRemake(prob.system, tspan, prob.timegrid)
    
    predictor = let remaker = remaker, odae_prob = odae_prob, alg = prob.alg, options = prob.diffeq_options
        (p) ->  sequential_solve(remaker, odae_prob, alg, p; options...)
    end
end

get_initial_variables(prob::OEDProblem) = generate_initial_variables(prob.system, prob.timegrid)


function Optimization.OptimizationProblem(prob::OEDProblem, AD::Optimization.ADTypes.AbstractADType, u0::ComponentVector = get_initial_variables(prob), p = SciMLBase.NullParameters(); integer_constraints::Bool = false,
    constraints = nothing, variable_type::Type{T} = Float64, kwargs...
    ) where T
    u0 = T.(u0)
    p = !isa(p, SciMLBase.NullParameters) ?  T.(p) : p
    
    # A simple predictor
    solver = build_predictor(prob)

    f_idxs = [is_fisher_state(xi) for xi in states(prob.system)]
    n = Val(Int(sqrt(2 * sum(f_idxs) + 0.25) - 0.5))

    # Our objective function
    objective = let solver = solver, criterion = prob.objective, idx = f_idxs, n = n
        (p, x) -> begin
            x, _ = solver(p) 
            F = _symmetric_from_vector(x[idx, end], n)
            criterion(F) 
        end
    end

    # Generate the constraints, if possible
    if isa(constraints, ConstraintsSystem)
        (_, cons), cons_lb, cons_ub = ModelingToolkit.generate_function(constraints, expression = Val{false})
    else
        cons = cons_lb = cons_ub = nothing
    end

    # Bounds based on the variables
    lb = T.(generate_variable_bounds(prob.system, prob.timegrid, true))
    ub = T.(generate_variable_bounds(prob.system, prob.timegrid, false))

    @assert all(lb .<= u0 .<= ub) "The initial variables are not within the bounds. Please check the input!"

    # No integers
    integrality = Bool.(u0 * 0) 
    
    # Integer support
    if integer_constraints
        integrality.controls .= true
        integrality.measurements .= true
    end

    # Might be useful
    syms = Symbol.(vcat(get_initial_conditions(prob.system), get_control_parameters(prob.system), get_measurement_function(prob.system)))
    
    # Declare the Optimization function 
    opt_f = OptimizationFunction(
        objective, AD; 
        syms = syms,
        cons = cons,
    )

    # Return the optimization problem
    OptimizationProblem(opt_f, u0, p, lb = lb, ub = ub, int = integrality, 
        lcons = cons_lb, ucons = cons_ub
    )
end