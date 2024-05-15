# Design of Experiments for Urethan

In this example we take a look at the reaction of Urethane from the educts Butanol and Isocyanate. The reaction scheme is

```math 
\begin{aligned}
A + B &\rightarrow C\\ 
A + C &\rightleftarrow D\\
3 A &\rightarrow E
\end{aligned} 
```
with reactants Isocyanate A, Butanol B, Urethane C, Allophanate D and Isocyanurate E.

We start by using [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) and DynamicOED.jl to model the system.

```@example urethan
using DynamicOED
using ModelingToolkit
using Optimization, OptimizationMOI, Ipopt
using Plots

const M    = [ 0.11911, 0.07412, 0.19323, 0.31234, 0.35733, 0.07806 ]
const rho  = [1095.0, 809.0, 1415.0, 1528.0, 1451.0, 1101.0]

Rg   = 8.314;
T1   = 363.16;

n_A0, n_B0, n_L0 = 0.1, 0.05, 0.01

# Set up variables
@variables t
D = Differential(t)
@variables h(t)[1:4] [description = "Observed", measurement_rate=10]
h = collect(h)

@variables n_C(t)=0.0 [description = "Molar numbers for C"]
@variables n_D(t)=0.0 [description = "Molar numbers for D"]
@variables n_E(t)=0.0 [description = "Molar numbers for E"]
@variables n_A(t)=n_A0 [description = "Molar numbers for C"]
@variables n_B(t)=n_B0 [description = "Molar numbers for D"]
@variables n_L(t)=n_L0 [description = "Molar numbers for E"]

@parameters begin
    p1=1.0, [description = "Scaling parameter 1", tunable = true]
    p2=1.0, [description = "Scaling parameter 2", tunable = true]
    p3=1.0, [description = "Scaling parameter 3", tunable = true]
    p4=1.0, [description = "Scaling parameter 4", tunable = false]
    p5=1.0, [description = "Scaling parameter 5", tunable = false]
    p6=1.0, [description = "Scaling parameter 6", tunable = false]
end

# Variables for temperature and feed
@variables feed1(t)=0 [description = "State for feed 1"]
@variables feed2(t)=0 [description = "State for feed 2"]
@variables temperature(t)=373.15 [description = "State for temperature"]

# Controls for temperature and feed
@parameters begin
    u1=0.0125, [description = "RHS of feed1", bounds=(0.0,0.0125), input=true, measurement_rate=10]
    u2=0.0125, [description = "RHS of feed2", bounds=(0.0,0.0125), input=true, measurement_rate=10]
    u_temp=0.0, [description = "RHS of temperature", bounds=(-15,15), input=false]
end


# Write system of equations
k_ref1    = p1 * 5.0E-4
E_a1      = p2 * 35240.0
k_ref2    = p3 * 8.0E-8
E_a2      = p4 * 85000.0
k_ref4    = p5 * 1.0E-8
E_a4      = p6 * 35000.0
dH_2      = -17031.0
K_C2      = 0.17

# Arrhenius equations for the reaction rates
fac_T = 1.0 / (Rg*temperature) - 1.0 / (Rg*T1);
k1 = k_ref1 * exp(- E_a1 * fac_T);
k2 = k_ref2 * exp(- E_a2 * fac_T);
k4 = k_ref4 * exp(- E_a4 * fac_T);
K_C = K_C2 * exp(- dH_2 * fac_T);
k3 = k2/K_C;

# Reaction volume
V  = n_A * M[1] / rho[1] + n_B * M[2] /rho[2] + n_C * M[3] / rho[3] + n_D * M[4] / rho[4] +
            n_E * M[5] / rho[5] + n_L * M[6] / rho[6]

# Reaction rates
r1 = k1 * n_A/V * n_B/V;
r2 = k2 * n_A/V * n_C/V;
r3 = k3 * n_D/V;
r4 = k4 * (n_A / V)*(n_A / V);

sum_observed = n_A * M[1]  + n_B * M[2]  + n_C * M[3]  + n_D * M[4]  + n_E * M[5] + n_L * M[6]
# Define the eqs
@named urethan = ODESystem(
    [
        D(feed1) ~ u1;
        D(feed2) ~ u2;
        D(temperature) ~ u_temp;
        D(n_C) ~ V * (r1 - r2 + r3); #n_C
        D(n_D) ~ V * (r2 - r3);      #n_D
        D(n_E) ~ V * r4;             #n_E
        0 ~ n_A0 + feed1 - n_C - 2n_D - 3n_E - n_A;  #n_A
        0 ~ n_B0 + feed2 - n_C - n_D - n_B;   #n_B
        0 ~ n_L0 + (feed1 + feed2) - n_L;     #n_L
    ],  tspan       = (0.0, 80.0),
        observed    = h .~ [100 * n_A*M[1]/sum_observed;
                            100 * n_C*M[3]/sum_observed;
                            100 * n_D*M[4]/sum_observed;
                            100 * n_E*M[5]/sum_observed
                           ]
)
```

Like in the [Design of Experiments for a simple system](@ref), we added important information:

- The observed variables are initialized with a [`measurement_rate`](@ref DynamicOED.VariableRate). This time we use an integer measurement rate, resulting in $10$ subintervals of equal length.
- The parameters $p_1$ t0 $p_3$ that enter the first and second Arrhenius equations are marked as `tunable`. These are the parameters we want to estimate.

Now we can build the `OEDSystem`, which includes the necessary equations for the sensitivities and the Fisher information matrix. For this, we only consider the differential equations of the species $C, D$ and $E$.

```@example urethan
relevant_equations = equations(urethan)[4:6]
relevant_states = states(urethan)[4:6]

@named oed = OEDSystem(urethan, equationset = relevant_equations, stateset = relevant_states)
```

With this augmented `ODESystem` we can set up the `OEDProblem` by specifying the criterion we want to optimize.

```@example urethan
oed_simp = structural_simplify(oed)
oed_problem = OEDProblem(oed_simp, ACriterion())
```
We choose the [`ACriterion`](@ref), which minimizes the trace of the inverse of the Fisher information matrix. For constraining the time we can measure by defining a `ConstraintSystem` from ModelingToolkit on the optimization variables. We allow measurements of each species in only 2 of the 10 subintervals. Also, we limit the amount of energy we give into the system via an upper limit on the sum of the temperature controls.

```@example urethan
optimization_variables = states(oed_problem)
w1, w2, w3, w4 = keys(optimization_variables.measurements)
u1, u2 = keys(optimization_variables.controls)

constraint_equations = [
      sum(optimization_variables.measurements[w1]) ≲ 2,
      sum(optimization_variables.measurements[w2]) ≲ 2,
      sum(optimization_variables.measurements[w3]) ≲ 2,
      sum(optimization_variables.measurements[w4]) ≲ 2,
]


@named constraint_system = ConstraintsSystem(
    constraint_equations, collect(optimization_variables), []
)
nothing # hide
```
!!! note 
    The `optimization_variables` contain several groups of variables, namely `measurements`, `controls`, `initial_conditions`, and `regularization`. `measurements` represent the decision to observe at a specific time point at the grid. We currently work with the naming convention `w_i` for the i-th observed equation. Currently we need to `collect` the states before passing them into the `ConstraintsSystem`!


Finally, we are now able to convert our [`OEDProblem`](@ref) into an `OptimizationProblem` and `solve` it.

!!! note 
    Currently we only support `AutoForwardDiff()` as an AD backend.


```@example urethan
optimization_problem = OptimizationProblem(
    oed_problem, AutoForwardDiff(), constraints = constraint_system,
    integer_constraints = false
)

optimal_design = solve(optimization_problem, Ipopt.Optimizer(); hessian_approximation="limited-memory")

u_opt = optimal_design.u + optimization_problem.u0
```

Now we want to visualize the found solution. 
```@example urethan

predictor = DynamicOED.build_predictor(oed_problem)
x_opt, t_opt = predictor(u_opt)
timegrid = oed_problem.timegrid

np = sum(istunable.(parameters(urethan)))
nx = length(relevant_equations)
sts = states(oed_simp)


states_plot1 = plot(t_opt, x_opt[4:5,:]', label=hcat(string.(sts[4:5])...), xlabel="Time", ylabel="Concentrations")
states_plot2 = plot(t_opt, x_opt[6,:], label=string(sts[6]), xlabel="Time", ylabel="Concentrations", color=3)

feed_plot = plot(t_opt, x_opt[1:2, :]', xlabel = "Time", ylabel = "Feed", label = hcat([string(x) for x in sts[1:2]]...))
temp_plot = plot(t_opt, x_opt[3, :], xlabel="Time", ylabel="Temperature", label=string(sts[3]))
hspan!([293, 473], alpha=.3, label=nothing)


states_plot1 = plot(t_opt, x_opt[4:5,:]', label=hcat(string.(sts[4:5])...), xlabel="Time", ylabel="Concentrations")
states_plot2 = plot(t_opt, x_opt[6,:], label=string(sts[6]), xlabel="Time", ylabel="Concentrations", color=3)

sens_vars = startswith.(string.(sts), "(G")

sensitivities_plot = plot(t_opt, x_opt[sens_vars,:]', label=hcat(string.(sts[sens_vars])...), legend_font_pointsize=6, legend_columns=np, xlabel="Time", ylabel="dx/dp")

u1_, u2_, = keys(optimization_variables.controls)

repfirst(x) = [x[1]; x]

control_feed_plot = plot(t_opt, repfirst(u_opt.controls[u1_]), label="u1(t)", xlabel="Time", linetype=:steppre)
plot!(t_opt, repfirst(u_opt.controls[u2_]), label="u2(t)", xlabel="Time", linetype=:steppre)

ws = keys(optimization_variables.measurements)
sampling_plot = plot()
for wi in ws
    w_i = u_opt.measurements[wi]
    plot!(t_opt, repfirst(w_i), linetype=:steppre, xlabel="Time", ylabel="Sampling", label=string(wi))
end



l = @layout [
    grid(2,3)
    a{0.3h}
    b{0.2h}
]

plot(feed_plot, temp_plot, states_plot1, control_feed_plot,  states_plot2, plot(), sensitivities_plot, sampling_plot, layout=l, size=(900,600))

```

