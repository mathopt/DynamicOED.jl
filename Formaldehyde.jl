using DynamicOED
using ModelingToolkit
using SciMLSensitivity
using Optimization, OptimizationMOI, Ipopt
using Plots
using OrdinaryDiffEq
using blockSQP

temperature = 40
R = 1.987204258640

#######################
# given parameter estimates
#######################

K₁ = 1624
K₂ = 10000
K₃ = 17.3
K₄ = 10^(-13.25)
K₅ = 10^(-10.1)
K₆ = 10^(-7.4)
K₇ = 10.e-12
Kw = 2.92e-14

# equilibrium constants
α₈ = 5.e10
E₈ = 14.5
α₉ = 1.e12
E₉ = 22.3
α₁₀ = 1.e12
E₁₀ = 20.1
kf = 7.37e-5
kr = 1.4e11

αo₈ = log(α₈)
Eo₈ = E₈

H₂O = 55

# rescale to Kelvin
temperature += 273.15

#######################
# initial parameter estimates
#######################

# eliminate y₁₁, y₁₄, and y₁₅
u0 = vcat(0.1,
          0.1,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.1,
          0.0,
          0.0)


tunable_start = false

@variables t
@variables y₁(t) = u0[1] [description = "CH₂O"]
@variables y₂(t) = u0[2] [description = "H₂O₂"] 
@variables y₃(t) = u0[3] [description = "HO-CH₂-OH"]
@variables y₄(t) = u0[4] [description = "HO-CH₂-O⁻"]
@variables y₅(t) = u0[5] [description = "HO-CH₂-OOH"]
@variables y₆(t) = u0[6] [description = "HO-CH₂-OO⁻"]
@variables y₇(t) = u0[7] [description = "HO-CH₂-OO-CH₂-OH"]
@variables y₈(t) = u0[8] [description = "HO-CH₂-OO-CH₂-O⁻"]
@variables y₉(t) = u0[9] [description = "⁻O-CH₂-OO-CH₂-O⁻"]
@variables y₁₀(t) = u0[10] [description = "H₂"]
@variables y₁₁(t) = u0[11] [description = "H⁺"]
@variables y₁₂(t) = u0[12] [description = "OH⁻"]
@variables y₁₃(t) = u0[13] [description = "CH₃OH"]
@variables y₁₄(t) = u0[14] [description = "HCOO⁻"]

@variables u(t) [description = "Control"]
# @parameters p_fix[1:length(p_fix_init)] = p_fix_init [description = "Fixed Parameters", tunable = false]
@parameters p_tune[1:2] = [αo₈, Eo₈] [description = "Tunable Parameters", tunable = true]
D = Differential(t)
@variables obs(t)[1:3] [description = "Observed", measurement_rate = 5]
obs = collect(obs)

# functions of tunable parameters
k₈  = exp(p_tune[1]) * exp(-p_tune[2] / (R * temperature))
k₉  = α₉ * exp(-E₉ / (R * temperature))
k₁₀ = α₉ * exp(-E₉ / (R * temperature))

# ODE equations
f₃ = -(k₉ + k₁₀) * y₃^2 * y₁₂
f₉ = -k₈ * y₉
f₁₀ = k₈ * y₉
f₁₂ = kf - kr * y₁₁ * y₁₂
f₁₃ = -0.5 * f₃
f₁₄ = k₈ * y₉

# algebraic equations
a₁ = -y₁ + y₃ / (K₁ * H₂O)
a₂ = -y₂ + y₅ / (K₂ * y₁)
a₄ = -y₄ + (K₄ * y₃) / y₈
a₅ = -y₅ + y₇ / (K₃ * y₁)
a₆ = -y₆ + (K₅ * y₅) / y₁₁
a₇ = -y₇ + (y₈ * y₁₁) / K₆
a₈ = -y₈ + (y₉ * y₁₁) / K₇
aw = y₁₁ * y₁₂ - Kw

@named dow_oed = ODESystem(
    [
        D(y₃) ~  f₃;
        D(y₉) ~  f₉;
        D(y₁₀) ~  f₁₀;
        D(y₁₂) ~  f₁₂;
        D(y₁₃) ~  f₁₃;
        D(y₁₄) ~  f₁₄;
        0 ~ a₁;
        0 ~ a₂;
        0 ~ a₄;
        0 ~ a₅;
        0 ~ a₆;
        0 ~ a₇;
        0 ~ a₈;
        0 ~ aw
    ], tspan = (0.0, 20.0),
    observed = obs .~ [y₁; y₁₃; y₁₄]
)

# simp_dow = structural_simplify(dow_oed)
sys = OEDSystem(dow_oed,name = :dow)

oed_problem = OEDProblem(sys, ACriterion(), alg = DFBDF(), sensealg = ForwardDiffSensitivity())

optimization_variables = states(oed_problem)

fullprob = DAEProblem(oed_problem.system, zeros(size(equations(sys))), vcat(u0, zeros(size(equations(sys),1) - size(u0,1))), (0, 10.0) )

# plot(fullsol, idxs=collect(9:21), legend=false)

constraint_equations = [
    sum(optimization_variables.measurements.w₁) ≲ 10,
    sum(optimization_variables.measurements.w₂) ≲ 10,
    sum(optimization_variables.measurements.w₃) ≲ 10,
]

@named constraint_system = ConstraintsSystem(
    constraint_equations, collect(optimization_variables), Num[]
)

optimization_problem = OptimizationProblem(
    oed_problem, AutoForwardDiff(), constraints = constraint_system,
    integer_constraints = false
)


optimization_problem.f(optimization_problem.u0, optimization_problem.p)
OrdinaryDiffEq.ForwardDiff.gradient(Base.Fix2(optimization_problem.f, optimization_problem.p), optimization_problem.u0)


optimal_design = solve(optimization_problem,  Ipopt.Optimizer(); hessian_approximation="limited-memory", max_iter = 50)
#optimal_design = solve(optimization_problem, BlockSQPOpt())

# 
mus = optimal_design.original.inner.mult_g
println(mus)
u_opt = optimal_design.u + zero(optimization_problem.u0)

function plotoed(problem, res)

    predictor = DynamicOED.build_predictor(problem)
    x_opt, t_opt = predictor(res)
    timegrid = problem.timegrid

    state_plot = plot(t_opt, x_opt[:, :]', xlabel = "Time", ylabel = "States")#, label = ["y₁" "y₂" "y₃" "y₄" "y₅" "y₆" "y₇" "y₈" "y₉" "y₁₀"])

    measures_plot = plot()
    for i in 1:3
        t_measures = vcat(first.(timegrid.timegrids[i]), last.(timegrid.timegrids[i]))
        sort!(t_measures)
        unique!(t_measures)
        _measurements = getfield(res.measurements |> NamedTuple, timegrid.variables[i])
        plot!(t_measures,
            vcat(_measurements, last(_measurements)),
            line = :steppost,
            xlabel = "Time",
            xticks = 0:10:200,
            ylabel = "Measurement",
            #color = i == 2 ? :red : :blue,
            label = string(timegrid.variables[i]))
    end

    plotd = plot(state_plot, measures_plot, layout=(2,1),size=(700,700), legend = :outertopright)
    savefig(plotd,"plot.png")
end

plotoed(oed_problem, u_opt)

function print_oed(problem, res)
    predictor = DynamicOED.build_predictor(problem)
    x_opt, t_opt = predictor(res)
    timegrid = problem.timegrid
    open("log_states.txt","a") do io
        for i in eachindex(x_opt[1,:])
            println(io, x_opt[1:4, i])
        end
    end
    open("log_control.txt","a") do io
        for i in 1:4
            t_measures = vcat(first.(timegrid.timegrids[i]), last.(timegrid.timegrids[i]))
            sort!(t_measures)
            unique!(t_measures)
            _measurements = getfield(res.measurements |> NamedTuple, timegrid.variables[i])
            println(io,_measurements)
            println(io,t_measures)
        end
    end
end

# print_oed(oed_problem, u_opt)

sols = DynamicOED.continuous_predict(oed_problem, u_opt)
Qs = DynamicOED.extract_Qs(oed_problem, u_opt, sols)
u_opt = optimal_design.u + zero(optimization_problem.u0)
F, local_inf_gain, global_inf_gain, np, sols = DynamicOED.compute_information_gain(oed_problem, u_opt);
t_ = unique(reduce(vcat, [sol.t for sol in sols]))

u_opt_2 = copy(u_opt)
u_opt_2.regularization = 0.0
F, local_inf_gain, global_inf_gain, np, sols = DynamicOED.compute_information_gain(oed_problem, u_opt_2);


w1 = u_opt.measurements.w₁
tspan_w1 = oed_problem.timegrid.timegrids[1][w1 .> 0.9]


using LinearAlgebra
infgain_h1 = tr.(global_inf_gain[1])
infgain_h2 = tr.(global_inf_gain[2])
infgain_h3 = tr.(global_inf_gain[3])
infgain_h4 = tr.(global_inf_gain[4])
using Plots
plot(t_, infgain_h1, xticks=0:10:200)
Plots.hline!(mus[1:1]/4, label="μ", color=:black, linestyle=:dash)
Plots.vspan!(vcat(first(first(tspan_w1)), last(last(tspan_w1))), alpha=0.4, color=:gray, label=nothing)

plot(t_, infgain_h2, xticks=0:10:200)
Plots.hline!(mus[2:2]/4)
plot(t_, infgain_h3)
Plots.hline!(mus[3:3]/4)
plot(t_, infgain_h4)
Plots.hline!(mus[4:4]/4)