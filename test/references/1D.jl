using OrdinaryDiffEq
using ModelingToolkit
using LinearAlgebra
using DynamicOED
using Test
using Optimization, OptimizationMOI, Ipopt, Juniper

const TEST_CRITERIA = [
    FisherACriterion(), FisherDCriterion(), FisherECriterion(),
    ACriterion(), DCriterion(), ECriterion(),
]

## Define the system
@variables t
@variables x(t)=1.0 [description = "State"]
@parameters p[1:1]=-2.0 [description = "Fixed parameter", tunable = true]
@variables obs(t) [description = "Observed", measurement_rate = 10]
D = Differential(t)

# Define the eqs
@named simple_system = ODESystem([
        D(x) ~ p[1] * x,
    ], tspan = (0.0, 1.0),
    observed = obs .~ [x])

@named oed = OEDSystem(simple_system)
oed = structural_simplify(oed)

@testset "Relaxed" begin
    optimizer = Ipopt.Optimizer()
    for (i, crit) in enumerate(TEST_CRITERIA)
        oed_problem = DynamicOED.OEDProblem(structural_simplify(oed), crit)

        optimization_variables = states(oed_problem)
        timegrids = DynamicOED.get_timegrids(oed_problem)

        # Convert to delta
        Δts = map(timegrids) do grid
            -map(x -> -(x...), grid)
        end

        constraints = [
            0 ≲ 0.2 .- 0.5 * sum(Δts.w₁ .* optimization_variables.measurements.w₁),
        ]

        # Define an MTK Constraint system
        @named constraint_set = ConstraintsSystem(constraints,
            reduce(vcat, optimization_variables),
            [])

        opt_prob = OptimizationProblem(oed_problem,
            AutoForwardDiff(),
            constraints = constraint_set,
            integer_constraints = false)
        res = solve(opt_prob, optimizer)

        @test isapprox(res.u[1:(end - 1)],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            atol = 1e-5,
            rtol = 1e-1)
        @test isapprox(res.objective,
            i <= 3 ? -1.2823515846720533e-02 : 2.0,
            atol = 1e-2,
            rtol = 1e-1)
    end
end

@testset "Integer" begin
    optimizer = OptimizationMOI.MOI.OptimizerWithAttributes(Juniper.Optimizer,
        "nl_solver" => OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer,
            "print_level" => 0))
    for (i, crit) in enumerate(TEST_CRITERIA)
        oed_problem = DynamicOED.OEDProblem(structural_simplify(oed), crit)

        optimization_variables = states(oed_problem)
        timegrids = DynamicOED.get_timegrids(oed_problem)

        # Convert to delta
        Δts = map(timegrids) do grid
            -map(x -> -(x...), grid)
        end

        constraints = [
            0 ≲ 0.2 .- 0.5 * sum(Δts.w₁ .* optimization_variables.measurements.w₁),
        ]

        # Define an MTK Constraint system
        @named constraint_set = ConstraintsSystem(constraints,
            reduce(vcat, optimization_variables),
            [])

        opt_prob = OptimizationProblem(oed_problem,
            AutoForwardDiff(),
            constraints = constraint_set,
            integer_constraints = true)
        res = solve(opt_prob, optimizer)

        @test isapprox(res.u[1:(end - 1)],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            atol = 1e-5,
            rtol = 1e-1)
        @test isapprox(res.objective,
            i <= 3 ? -1.2823515846720533e-02 : 2.0,
            atol = 1e-2,
            rtol = 1e-1)
    end
end
