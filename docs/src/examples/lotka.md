# Design of Experiments for Lotka-Volterra

Now, let us consider the well-known Lotka-Volterra system  

```math 
\begin{aligned}
\dot{x_1} &= p_1 x - p_2 x y\\ 
\dot{x_2} &= - p_3 y + p_4 x y\\
y(x)      &= x
\end{aligned} 
```

where we are interested in estimating the parameters $p_2$ and $p_4$. We can measure the states directly with some fixed measurement rate.

We start by using [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) and DynamicOED.jl to model the system.

```@example lotka
using DynamicOED
using ModelingToolkit
using Optimization, OptimizationMOI, Ipopt
using Plots

@variables t
@variables x(t)=0.5 [description = "Biomass Prey"] y(t)=0.7 [description ="Biomass Predator"]
@variables u(t) [description = "Control"]
@parameters p[1:2]=[1.0; 1.0] [description = "Fixed Parameters", tunable = false]
@parameters p_est[1:2]=[1.0; 1.0] [description = "Tunable Parameters", tunable = true]
D = Differential(t)
@variables obs(t)[1:2] [description = "Observed", measurement_rate = 96]
obs = collect(obs)

@named lotka_volterra = ODESystem(
    [
        D(x) ~   p[1]*x - p_est[1]*x*y;
        D(y) ~  -p[2]*y + p_est[2]*x*y
    ], tspan = (0.0, 12.0),
    observed = obs .~ [x; y]
)
```

Like in the [Design of Experiments for a simple system](@ref), we added important information:

- The observed variables are initialized with a [`measurement_rate`](@ref DynamicOED.VariableRate). This time we use an integer measurement rate, resulting in $96$ subintervals of equal length.
- The parameters $p_2$ and $p_4$ are marked as `tunable`. 

Now we can augment the system with the needed expressions for the sensitivities and the Fisher Information matrix by constructing an `OEDSystem`. 

```@example lotka
@named oed_system = OEDSystem(lotka_volterra)
```

With this augmented `ODESystem` we can set up the `OEDProblem` by specifying the criterion we want to optimize.

```@example lotka
oed_problem = OEDProblem(structural_simplify(oed_system), DCriterion())
```
We choose the [`DCriterion`](@ref), which minimizes the determinant of the inverse of the Fisher Information matrix. For constraining the time we can measure by defining a `ConstraintSystem` from ModelingToolkit on the optimization variables. We want to measure for at most $4$ units of time. Since we discretized the observed variables on $96$ subintervals on a time horizon of $12$ units of time, this translates to an upper limit on the measurements of $32$.

```@example lotka
optimization_variables = states(oed_problem)

constraint_equations = [
    sum(optimization_variables.measurements.w₁) ≲ 32,
    sum(optimization_variables.measurements.w₂) ≲ 32,
]

@named constraint_system = ConstraintsSystem(
    constraint_equations, optimization_variables, []
)
nothing # hide
```
!!! note 
    The `optimization_states` contain several groups of variables, namely `measurements`, `controls`, `initial_conditions`, and `regularization`. `measurements` represent the decision to observe at a specific time point at the grid. We currently work with the naming convention `w_i` for the i-th observed equation.


Finally, we are now able to convert our [`OEDProblem`](@ref) into an `OptimizationProblem` and `solve` it.

!!! note 
    Currently we only support `AutoForwardDiff()` as an AD backend.


```@example lotka
optimization_problem = OptimizationProblem(
    oed_problem, AutoForwardDiff(), constraints = constraint_system,
    integer_constraints = false
)

optimal_design = solve(optimization_problem, Ipopt.Optimizer(); hessian_approximation="limited-memory")

u_opt = optimal_design.u + optimization_problem.u0
```

Now we want to visualize the found solution. 
```@example lotka
function plotoed(problem, res)

    predictor = DynamicOED.build_predictor(problem)
    x_opt, t_opt = predictor(res)
    timegrid = problem.timegrid

    state_plot = plot(t_opt, x_opt[1:2, :]', xlabel = "Time", ylabel = "States", label = ["x" "y"])

    measures_plot = plot()
    for i in 1:2
        t_measures = vcat(first.(timegrid.timegrids[i]), last.(timegrid.timegrids[i]))
        sort!(t_measures)
        unique!(t_measures)
        _measurements = getfield(res.measurements |> NamedTuple, timegrid.variables[i])
        plot!(t_measures,
            vcat(_measurements, last(_measurements)),
            line = :steppost,
            xlabel = "Time",
            ylabel = "Measurement",
            color = i == 2 ? :red : :blue,
            label = string(timegrid.variables[i]))
    end

    plot(state_plot, measures_plot, layout=(2,1))
end

plotoed(oed_problem, u_opt)
```

