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
        measurement_rate = 10
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
        DCriterion(),
        alg = Rodas5(),
        diffeq_options = (; abstol = 1e-8, reltol = 1e-8))

    optimization_variables = states(oed_problem)
    @test length(optimization_variables) == 11

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
    @test isapprox(u_opt.measurements,
        [
            1.8872186300229836e-5,
            2.8175824999915748e-5,
            4.035472649222709e-5,
            5.881065041987711e-5,
            9.15959369452527e-5,
            0.0001667131402021681,
            0.0004865755450836175,
            0.9994267375362046,
            0.9997958106584781,
            0.9998709072042323
        ],
        atol = 1e-2,
        rtol = 1e-5)
    @test isapprox(u_opt[end], 0.0, atol = 1e-3)
    @test isapprox(res.objective, 0.0, atol = 1e-2)
end
