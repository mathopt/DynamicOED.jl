using OrdinaryDiffEq
using ModelingToolkit
using LinearAlgebra
using DynamicOED
using Test
using Optimization, OptimizationMOI, Ipopt, Juniper

@testset "Relaxed" begin
    @variables t [description = "Time"]
    @variables x(t)=0.5 [description = "Biomass Prey"] y(t)=0.7 [
        description = "Biomass Predator",
    ]
    @parameters p[1:4]=[1.0; 1.0; 0.4; 0.6] [
        description = "Fixed parameters",
        tunable = false,
    ]
    @parameters c[1:2]=[1.0; 1.0] [description = "Uncertain parameters", tunable = true]
    @parameters u=0.0 [
        description = "Binary control variable, measured 10 times over the provided time span, relaxed",
        measurement_rate = 0.3,
        input = true,
        bounds = (0.0, 1.0),
    ]
    @variables obs(t)[1:2] [
        description = "Observed variable measured 10 times over the provided time span",
        measurement_rate = 0.1,
    ]
    obs = collect(obs)
    D = Differential(t)

    # Define the ODE System
    @named lotka_volterra = ODESystem([D(x) ~ p[1] * x - c[1] * x * y - p[3] * x * u;
            D(y) ~ -p[2] * y + c[2] * x * y - p[4] * y * u], tspan = (0.0, 3.0),
        observed = obs .~ [x; y])

    @named oed_lotka = OEDSystem(lotka_volterra)
    oed_lotka = structural_simplify(oed_lotka)

    optimizer = Ipopt.Optimizer()

    oed_problem = DynamicOED.OEDProblem(oed_lotka, FisherACriterion())

    optimization_variables = states(oed_problem)
    timegrids = DynamicOED.get_timegrids(oed_problem)

    # Convert to delta
    Δts = map(timegrids) do grid
        -map(x -> -(x...), grid)
    end

    constraints = [
        0 ≲ 0.2 .- 0.5 * sum(Δts.w₁ .* optimization_variables.measurements.w₁),
        0 ≲ 0.2 .- 0.5 * sum(Δts.w₁ .* optimization_variables.measurements.w₂),
        1.0 ≳ sum(Δts.u .* optimization_variables.controls.u),
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
    u_opt = res.u + opt_prob.u0
    
    @test isapprox(u_opt.controls.u, [
        0.62,
        -4.357849235877129e-9,
        -7.0697653556629854e-9,
        -7.964662383759068e-9,
        -8.397838943460663e-9,
        -8.638244691196196e-9,
        -8.770673425035107e-9,
        -8.822579849602493e-9,
        -8.756712678420366e-9,
        -7.15514843922874e-9,
    ], atol = 1e-2, rtol = 1e-5)

    @test isapprox(u_opt.measurements.w₁, [
        -5.7627792545520196e-9,
        -5.758386540897064e-9,
        -5.749649687429022e-9,
        -5.736178435704279e-9,
        -5.717124027726658e-9,
        -5.691681964890273e-9,
        -5.65870824490519e-9,
        -5.616742454158438e-9,
        -5.563930946052556e-9,
        -5.497919260878003e-9,
        -5.415700919057426e-9,
        -5.313401639570466e-9,
        -5.185965416275459e-9,
        -5.026687451006144e-9,
        -4.826499153911781e-9,
        -4.572836596939433e-9,
        -4.247778691605136e-9,
        -3.824834680528813e-9,
        -3.2630718940717005e-9,
        -2.4955853136454706e-9,
        -1.4046890962909512e-9,
        2.383092076856661e-10,
        2.94606275311693e-9,
        8.15589586241535e-9,
        2.195770485198414e-8,
        2.997362515384574e-7,
        0.9999999700398273,
        0.9999999928632054,
        0.9999999994598789,
        1.000000002696486,
    ], atol = 1e-2, rtol = 1e-5)

    @test isapprox(u_opt.measurements.w₂ , [
        -3.833322378832553e-9,
        -3.829263549940177e-9,
        -3.821863054784994e-9,
        -3.811364308517609e-9,
        -3.7975638124861675e-9,
        -3.780429054546319e-9,
        -3.7596396952104195e-9,
        -3.734674487401935e-9,
        -3.7047719081855754e-9,
        -3.6688681558681895e-9,
        -3.6255041728703425e-9,
        -3.572687525940649e-9,
        -3.507686112897051e-9,
        -3.4267156244994873e-9,
        -3.3244548350663984e-9,
        -3.193271082440126e-9,
        -3.0219360290464985e-9,
        -2.7933954623011895e-9,
        -2.48066819075095e-9,
        -2.038742338841465e-9,
        -1.387008539910998e-9,
        -3.661772722445571e-10,
        1.387449921834108e-9,
        4.922025446167527e-9,
        1.4987797194431175e-8,
        2.4620247949157034e-7,
        0.9999999865101933,
        0.9999999999387468,
        1.0000000041944848,
        1.0000000061707939,
    ], atol = 1e-2, rtol = 1e-5)

    @info res.objective
    @test isapprox(res.objective, -4.85, atol = 1e-2)
end

@testset "Integer" begin
    @variables t [description = "Time"]
    @variables x(t)=0.5 [description = "Biomass Prey"] y(t)=0.7 [
        description = "Biomass Predator",
    ]
    @parameters p[1:4]=[1.0; 1.0; 0.4; 0.6] [
        description = "Fixed parameters",
        tunable = false,
    ]
    @parameters c[1:2]=[1.0; 1.0] [description = "Uncertain parameters", tunable = true]
    @parameters u::Int=0 [
        description = "Binary control variable, measured 10 times over the provided time span, discrete",
        measurement_rate = 0.3,
        input = true,
        bounds = (0.0, 1.0),
    ]
    @variables obs(t)[1:2] [
        description = "Observed variable measured 10 times over the provided time span",
        measurement_rate = 0.1,
    ]
    obs = collect(obs)
    D = Differential(t)

    # Define the ODE System
    @named lotka_volterra = ODESystem([D(x) ~ p[1] * x - c[1] * x * y - p[3] * x * u;
            D(y) ~ -p[2] * y + c[2] * x * y - p[4] * y * u], tspan = (0.0, 3.0),
        observed = obs .~ [x; y])

    @named oed_lotka = OEDSystem(lotka_volterra)
    oed_lotka = structural_simplify(oed_lotka)

    optimizer = OptimizationMOI.MOI.OptimizerWithAttributes(Juniper.Optimizer,
        "nl_solver" => OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer,
            "print_level" => 0))

    oed_problem = DynamicOED.OEDProblem(oed_lotka, DCriterion())

    optimization_variables = states(oed_problem)
    timegrids = DynamicOED.get_timegrids(oed_problem)

    # Convert to delta
    Δts = map(timegrids) do grid
        -map(x -> -(x...), grid)
    end

    constraints = [
        0 ≲ 0.2 .- 0.5 * sum(Δts.w₁ .* optimization_variables.measurements.w₁),
        0 ≲ 0.2 .- 0.5 * sum(Δts.w₁ .* optimization_variables.measurements.w₂),
        2.0 ≳ sum(optimization_variables.controls.u),
        sum(optimization_variables.controls.u) ≳ 1.0,
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
    u_opt = res.u + opt_prob.u0
    @test isapprox(u_opt.controls.u, [
        1.0000000032660439,
        1.6752611729303576e-8,
        -3.957801293389325e-9,
        -6.502972649205149e-9,
        -7.427886841755683e-9,
        -7.867272354160097e-9,
        -8.072531050877907e-9,
        -8.10418072743497e-9,
        -7.828539249999535e-9,
        2.2919106922838742e-8,
    ], atol = 1e-2, rtol = 1e-5)
    @test isapprox(u_opt.measurements.w₁, [
        -5.211901990293106e-9,
        -5.207076857082087e-9,
        -5.198756751528438e-9,
        -5.186616732374356e-9,
        -5.168953238707909e-9,
        -5.145233000351564e-9,
        -5.114238808971629e-9,
        -5.074425231655302e-9,
        -5.0238319061150834e-9,
        -4.9599686397941515e-9,
        -4.8796485940419125e-9,
        -4.77875169967176e-9,
        -4.651884100988051e-9,
        -4.491863965889369e-9,
        -4.288932171877283e-9,
        -4.029500707162183e-9,
        -3.6940706984933965e-9,
        -3.253610140730972e-9,
        -2.662867389692508e-9,
        -1.847044858368218e-9,
        -6.725891558394106e-10,
        1.1255766863260704e-9,
        4.161258552283267e-9,
        1.0253690769212536e-8,
        2.8070321989895634e-8,
        2.669729642217363e-7,
        0.9999999785713451,
        0.9999999930680564,
        1.0000000001481675,
        1.0000000033846281,
    ], atol = 1e-2, rtol = 1e-5)

    @test isapprox(u_opt.measurements.w₂, [
        -3.019543245725386e-9,
        -3.0157278940743817e-9,
        -3.0099643773066394e-9,
        -3.0023742929614693e-9,
        -2.9919714911197447e-9,
        -2.979104467416286e-9,
        -2.963502021727414e-9,
        -2.9447363599411898e-9,
        -2.9221929828099556e-9,
        -2.895009176783671e-9,
        -2.8619995687624747e-9,
        -2.8215415611469203e-9,
        -2.771395105562318e-9,
        -2.708443180301639e-9,
        -2.6282872894552796e-9,
        -2.5245964973797754e-9,
        -2.388053287302493e-9,
        -2.204547028931154e-9,
        -1.9519098271450428e-9,
        -1.5936619235443725e-9,
        -1.065948726823686e-9,
        -2.471065074110847e-10,
        1.1246260478619954e-9,
        3.728254842123207e-9,
        1.000256431978952e-8,
        2.530913016591332e-7,
        0.999999969264283,
        1.000000000489115,
        1.000000004387359,
        1.0000000064340413,
    ], atol =1e-2, rtol = 1e-5)
    @test isapprox(res.objective, 0.34, atol = 1e-2)
end

@testset "Relaxed with unknown initial condition" begin
    @variables t [description = "Time"]
    @variables x(t)=0.49 [
        description = "Biomass Prey",
        tunable = true,
        bounds = (0.1, 1.0),
    ] y(t)=0.7 [description = "Biomass Predator", tunable = false]
    @parameters p[1:3]=[1.0; 1.0; 1.0] [description = "Fixed parameters", tunable = false]
    @parameters c[1:1]=[1.0;] [description = "Uncertain parameters", tunable = true]
    @variables obs(t)[1:1] [
        description = "Observed variable measured 10 times over the provided time span",
        measurement_rate = 10,
    ]
    obs = collect(obs)
    D = Differential(t)

    # Define the ODE System
    @named lotka_volterra = ODESystem([D(x) ~ p[1] * x - c[1] * x * y;
            D(y) ~ -p[2] * y + p[3] * x * y], tspan = (0.0, 5.0),
        observed = obs .~ [y + x])

    @named oed_lotka = OEDSystem(lotka_volterra)
    oed_lotka = structural_simplify(oed_lotka)

    oed_problem = DynamicOED.OEDProblem(oed_lotka, DCriterion())

    optimization_variables = states(oed_problem)

    timegrids = DynamicOED.get_timegrids(oed_problem)

    # Convert to delta
    Δts = map(timegrids) do grid
        -map(x -> -(x...), grid)
    end

    constraints = [
        0 ≲ 1 .-  sum(Δts.w₁ .* optimization_variables.measurements.w₁),
    ]

    # Define an MTK Constraint system
    @named constraint_set = ConstraintsSystem(constraints,
        reduce(vcat, optimization_variables),
        [])

    opt_prob = OptimizationProblem(oed_problem,
        AutoForwardDiff(),
        constraints = constraint_set,
        integer_constraints = false)
    res = solve(opt_prob, Ipopt.Optimizer())
    u_opt = res.u + zero(opt_prob.u0)
    @test isapprox(u_opt.initial_conditions[1], 0.15, atol = 1e-2, rtol = 1e-5)
    @test isapprox(u_opt.measurements, [2.0786692724286787e-6, 2.0996128083833926e-6, 2.1369860754014572e-6, 2.215090580904563e-6, 2.3973894468800598e-6, 2.8899305755059397e-6, 4.771191106750985e-6, 7.588080619413368e-5, 0.9999093339443416, 0.9999941524754293], atol = 1e-2, rtol = 1e-5)
    @test isapprox(res.objective, 1e-2, atol = 1e-2, rtol = 1e-5)
end
