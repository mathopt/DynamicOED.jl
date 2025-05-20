using DynamicOED
using ModelingToolkit
using SciMLSensitivity
using Optimization, OptimizationMOI, Ipopt
using Plots
using OrdinaryDiffEq
using blockSQP

temperature = 40
T₀ = 69 + 273.15
R = 1.987204258640
Q = 0.0131

#######################
# given starting values for the first four states
#######################
if temperature == 40
    states_init = [1.7066, 8.32, 0.01, 0]
elseif temperature == 67
    states_init = [1.6749, 8.2262, 0.0104, 0.0017]
else
    states_init = [1.5608, 8.3546, 0.0082, 0.0086]
end

# rescale to Kelvin
temperature += 273.15

#######################
# initial parameter estimates
#######################

p_init = [1.3978e12,3.1760e12,4.8880e28,1.847e4,1.882e4,2.636e4,1.4681e-17,6.4115e-7,1.e-17]

function par_transform(p)
    R = 1.987204258640
    T₀ = 69 + 273.15

    k₀₁  = log(p[1] * exp(-p[4] / (R * T₀)))
    k₀₂  = log(p[2] * exp(-p[5] / (R * T₀)))
    k₀₋₁ = log(p[3] * exp(-p[6] / (R * T₀)))

    E₀₁  = p[4] / 1.e4
    E₀₂  = p[5] / 1.e4
    E₀₋₁ = p[6] / 1.e4

    lnK₁  = log(p[7])
    lnK₂  = log(p[8])
    lnK₋₁ = log(p[9])

    new_p = [k₀₁, k₀₂, k₀₋₁, E₀₁, E₀₂, E₀₋₁, lnK₁, lnK₂, lnK₋₁]
    return new_p
end

u0 = vcat(states_init,
          0.0,
          0.0131,
          0.5 * (-p_init[8] + sqrt(p_init[8]^2 + 4 * p_init[8] *states_init[1])),
          0.5 * (-p_init[8] + sqrt(p_init[8]^2 + 4 * p_init[8] * states_init[1])),
          0.0,
          0.0)

# select for each parameter whether it shall be estimated or fixed
p_tranformed = par_transform(p_init)
estimation_choice = [1, 0, 0, 1, 0, 0, 0, 0, 0]
p_fix_init = []
p_tune_init = []
for i in eachindex(estimation_choice)
    global p_tranformed, p_fix_init, p_tune_init
    curr_par = p_tranformed[i]
    if estimation_choice[i] == 1
        p_tune_init = vcat(p_tune_init, curr_par)
    else
        p_fix_init = vcat(p_fix_init, curr_par)
    end
end

tunable_start = false

@variables t
if tunable_start
    @variables y₁(t) = states_init[1] [description = "Variable 1", tunable = true, bounds = (states_init[1] * 0.5, states_init[1] * 2)]
    @variables y₂(t) = states_init[2] [description = "Variable 2", tunable = true, bounds = (states_init[2] * 0.5, states_init[2] * 2)]
    @variables y₃(t) = states_init[3] [description = "Variable 3", tunable = true, bounds = (states_init[3] * 0.5, states_init[3] * 2)]
    @variables y₄(t) = states_init[4] [description = "Variable 4", tunable = true, bounds = (states_init[4] * 0.5, states_init[4] * 2)]
else
    @variables y₁(t) = states_init[1] [description = "Variable 1"]
    @variables y₂(t) = states_init[2] [description = "Variable 2"]
    @variables y₃(t) = states_init[3] [description = "Variable 3"]
    @variables y₄(t) = states_init[4] [description = "Variable 4"]
end

@variables y₅(t) = 0. [description = "Variable 5"]
@variables y₆(t) = 0.0131 [description = "Variable 6"]
@variables y₇(t) = 0.5 * (-p_init[8] + sqrt(p_init[8]^2 + 4*p_init[8]*states_init[1])) [description = "Variable 7"]
@variables y₈(t) = 0.5 * (-p_init[8] + sqrt(p_init[8]^2 + 4*p_init[8]*states_init[1])) [description = "Variable 8"]
@variables y₉(t) = 0. [description = "Variable 9"]
@variables y₁₀(t) = 0. [description = "Variable 10"]

@variables u(t) [description = "Control"]
@parameters p_fix[1:length(p_fix_init)] = p_fix_init [description = "Fixed Parameters", tunable = false]
@parameters p_tune[1:length(p_tune_init)] = p_tune_init [description = "Tunable Parameters", tunable = true]
D = Differential(t)
@variables obs(t)[1:4] [description = "Observed", measurement_rate = 50]
obs = collect(obs)

# summarize all parameters
p_est = []
tune_counter, fix_counter = 1, 1
for i in eachindex(estimation_choice)
    global p_fix, p_tune, p_est
    global tune_counter, fix_counter
    if estimation_choice[i] == 1
        p_est = vcat(p_est, p_tune[tune_counter])
        tune_counter += 1
    else
        p_est = vcat(p_est, p_fix[fix_counter])
        fix_counter += 1
    end
end

print("p_est: ", p_est)

# functions of tunable parameters
k₁  = exp(p_est[1]) * exp(-p_est[4] * 1.e4/(R) * (1/temperature - 1/T₀))
k₂  = exp(p_est[2]) * exp(-p_est[5] * 1.e4/(R) * (1/temperature - 1/T₀))
k₋₁ = exp(p_est[3]) * exp(-p_est[6] * 1.e4/(R) * (1/temperature - 1/T₀))

# abbreviation of ODE
f₁ = -k₂ * y₈ * y₂
f₂ = -k₁ * y₆ * y₂ + k₋₁ * y₁₀ - k₂ * y₈ * y₂
f₃ = k₂ * y₈ * y₂ + k₁ * y₆ * y₄ - 0.5 * k₋₁ * y₉
f₄ = -k₁ * y₆ * y₄ + 0.5 * k₋₁ * y₉
f₅ = k₁ * y₆ * y₂ - k₋₁ * y₁₀
f₆ = -k₁ * (y₆ * y₂ + y₆ * y₄) + k₋₁ * (y₁₀ + 0.5 * y₉)

@named dow_oed = ODESystem(
    [
        D(y₁) ~  f₁;
        D(y₂) ~  f₂;
        D(y₃) ~  f₃;
        D(y₄) ~  f₄;
        D(y₅) ~  f₅;
        D(y₆) ~  f₆;
        0 ~ - y₇ - Q + y₆ + y₈ + y₉ + y₁₀;
        0 ~ - y₈ + (exp(p_est[8]) * y₁) / (exp(p_est[8]) + y₇);
        0 ~ - y₉ + (exp(p_est[9]) * y₃) / (exp(p_est[9]) + y₇);
        0 ~ - y₁₀ + (exp(p_est[7]) * y₅) / (exp(p_est[7]) + y₇)
    ], tspan = (0.0, 200.0),
    observed = obs .~ [y₁; y₂; y₃; y₄]
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
    sum(optimization_variables.measurements.w₄) ≲ 10,
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

    state_plot = plot(t_opt, x_opt[1:4, :]', xlabel = "Time", ylabel = "States")#, label = ["y₁" "y₂" "y₃" "y₄" "y₅" "y₆" "y₇" "y₈" "y₉" "y₁₀"])

    measures_plot = plot()
    for i in 1:4
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

print_oed(oed_problem, u_opt)

sols = DynamicOED.continuous_predict(oed_problem, u_opt)
Qs = DynamicOED.extract_Qs(oed_problem, u_opt, sols)
u_opt = optimal_design.u + zero(optimization_problem.u0)
F, local_inf_gain, global_inf_gain, np, sols = DynamicOED.compute_information_gain(oed_problem, u_opt);
t_ = unique(reduce(vcat, [sol.t for sol in sols]))

u_opt_2 = copy(u_opt)
u_opt_2.regularization = 1.0
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