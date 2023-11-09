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


function grid_variables(prob::OEDProblem)
    grid = prob.timegrid
    vars = grid.variables
    grid_vars = []
    @inbounds for i in eachindex(vars)
        local_vars = zeros(Num, axes(grid.timegrids[i], 1))
        for j in eachindex(local_vars)
            local_vars[j] += Symbolics.variable(vars[i], j)
        end
        push!(grid_vars, (vars[i], local_vars))
    end
    return sortkeys(NamedTuple(grid_vars)) |> ComponentVector
end


function build_predictor(prob::OEDProblem)
    tspan = (first(first(prob.timegrid.timespans)), last(last(prob.timegrid.timespans)))
    odae_prob = ODAEProblem(prob.system, Pair[], tspan)
    remaker = OEDRemake(prob.system, tspan, prob.timegrid)
    
    predictor = let remaker = remaker, odae_prob = odae_prob, alg = prob.alg, options = prob.diffeq_options
        (p) ->  sequential_solve(remaker, odae_prob, alg, p; options...)
    end
end


const NOCONSTRAINTS = ConstraintsSystem([], [], [], name = :no_constraints)


function Optimization.OptimizationProblem(prob::OEDProblem, AD::Optimization.ADTypes.AbstractADType; integer_constraints::Bool = false,
    terminal_constraints = NOCONSTRAINTS, grid_constraints = NOCONSTRAINTS, kwargs...
    )
    
    
    solver = build_predictor(prob)

    f_idxs = [is_fisher_state(xi) for xi in states(prob.system)]
    n = Val(Int(sqrt(2 * sum(f_idxs) + 0.25) - 0.5))

    objective = let solver = solver, criterion = prob.objective, idx = f_idxs, n = n
        (p, x) -> begin
            x, _ = solver(p) 
            F = _symmetric_from_vector(x[idx, end], n)
            criterion(F) 
        end
    end

    # Generate the constraints
    (_, terminal_constraints_f), terminal_lb, terminal_ub = ModelingToolkit.generate_function(terminal_constraints, expression = Val{false})
    (_, grid_constraints_f), grid_lb, grid_ub = ModelingToolkit.generate_function(grid_constraints, expression = Val{false})

    # Merge the constraints
    cons_lb = vcat(terminal_lb, grid_lb)
    cons_ub = vcat(terminal_ub, grid_ub)

    # Now build the overall function 
    cons = let terminal_constraints_f = terminal_constraints_f, grid_constraints_f = grid_constraints_f, N_terminals = size(terminal_lb, 1), solver = solver
        (res, p, x) -> begin
            grid_constraints_f(res[1:N_terminals], p, x)
            states, _ = solver(p)
            terminal_constraints_f(res[N_terminals+1:end], states[:, end], x)
            return res
        end
    end

    p0 = Float64.(generate_initial_variables(prob.system, prob.timegrid))
    lb = Float64.(generate_variable_bounds(prob.system, prob.timegrid, true))
    ub = Float64.(generate_variable_bounds(prob.system, prob.timegrid, false))

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
        lcons = cons_lb, ucons = cons_ub
    )
end