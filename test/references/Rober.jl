using OrdinaryDiffEq
using ModelingToolkit
using LinearAlgebra
using DynamicOED
using Test
using Optimization, OptimizationMOI, Ipopt, Juniper

@testset "Relaxed" begin
    @variables t y₁(t)=1.0 y₂(t)=0.0 y₃(t)=0.0
    @variables obs(t) [
        description = "Observed variable measured 10 times over the provided time span",
        measurement_rate = 10,
    ]
    @parameters k₁=0.04 [tunable = true]
    @parameters k₂=3e7 k₃=1e4
    D = Differential(t)
    eqs = [D(y₁) ~ -k₁ * y₁ + k₃ * y₂ * y₃
        D(y₂) ~ k₁ * y₁ - k₃ * y₂ * y₃ - k₂ * y₂^2
        0 ~ y₁ + y₂ + y₃ - 1]

    @named roberdae = ODESystem(eqs, tspan = (0, 100.0), observed = obs .~ [y₁])
    @named roberoed = OEDSystem(roberdae)
    roberoed = structural_simplify(dae_index_lowering(roberoed))

    optimizer = Ipopt.Optimizer()

    oed_problem = DynamicOED.OEDProblem(roberoed,
        FisherDCriterion(),
        alg = Rodas5(),
        diffeq_options = (; abstol = 1e-8, reltol = 1e-8))

    optimization_variables = states(oed_problem)
    @test length(optimization_variables) == 10

    constraints = [
        1.0 ≲ sum(optimization_variables.measurements.w₁),
        3.0 ≳ sum(optimization_variables.measurements.w₁)]

    # Define an MTK Constraint system
    @named constraint_set = ConstraintsSystem(constraints,
        reduce(vcat, optimization_variables),
        [])

    opt_prob = OptimizationProblem(oed_problem,
        AutoForwardDiff(),
        constraints = constraint_set,
        integer_constraints = false)
    res = solve(opt_prob, optimizer)
    u_opt = res.u + zero(opt_prob.u0)

    @test u_opt.measurements.w₁ ≈ [
        -9.964811377982917e-9,
        -9.94689705666124e-9,
        -9.922856160437833e-9,
        -9.885103640955005e-9,
        -9.813833309012011e-9,
        -9.625985945410416e-9,
        4.0100656295094836e-8,
        1.0000000094881993,
        1.000000009734404,
        1.000000009807539,
    ]
    @test isapprox(res.objective, -740.5907947052916, atol = 1e-2, rtol = 1e-2)
end
