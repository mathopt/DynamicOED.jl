---
title: 'NeuralOED.jl: A Julia package for solving dynamic optimum experimental design problems'
tags:
  - Julia
  - optimization
  - experimental design
  - parameter estimation
authors:
  - name: Julius Martensen
    #equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    orcid: 0000-0003-4143-3040
    corresponding : true
    affiliation: 1
  - name: Christoph Plate
    orcid: 0000-0003-0354-8904
    #equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Sebastian Sager
    orcid : 0000-0002-0283-9075 
    affiliation: 1
affiliations:
 - name: Otto von Guericke University Magdeburg, Germany
   index: 1
date: 14 February 2023
bibliography: bibliography.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Optimum experimental design (OED) problems are typically encountered when unknown or uncertain
parameters of mathematical models are to be estimated from experimental data. 
In this scenario, OED helps to decide on an experimental setup, i.e., deciding on how to control the process and
when to measure in order to maximize the amount of information gathered such that the parameters can be accurately estimated. 

Solving OED problems is of interest for several reasons. First, all model-based optimization strategies rely on the knowledge of the accurate values of the model's parameters. Second, 
it allows to reduce the number of experiments or measurements in practical applications, which is helpful when measuring quantities of interest is only possible to a limited extent, e.g., due to high costs of measurements.
Following ideas presented in [@Sager2013], we cast the OED problem into an optimal control 
problem. This can be done by augmenting the user-provided system of ordinary differential equations (ODE) or differential algebraic equations (DAE) with their variational differential (algebraic) equations and the differential equation governing the evolution of the Fisher information matrix (FIM). A suitable criterion based on the FIM is then optimized in the resulting control problem employing a direct *first discretize, then optimize* single shooting approach with standard NLP solvers.
For more information on optimal experimental design for DAEs and their sensitivity analysis, we refer to [@Li2000SensitivityAnalysisDifferential; @Koerkel2002]. 

# Statement of need

`DynamicOED.jl` is a Julia [@bezanson2017julia] package for solving optimum experimental 
design problems. It was designed to be used by researchers and modelers to easily investigate and analyze models and be able to collect insightful data for parameter estimation of dynamical systems. To our knowledge, it is the first package for solving dynamic optimal experimental design problems with differential equation models in the Julia programming language.

\pagebreak

# Problem statement 

The problem we are interested in solving reads

$$
\begin{array}{clcll}
\displaystyle \min_{x, G, F, z, w, u, x_0} 
& \multicolumn{3}{l}{ \phi(F(t_f))} &  \\[1.5ex]
\mbox{s.t.}     & 0  & = & f(\dot x(t), x(t), u(t), p) \\  
                & 0  & = & f_{\dot x}(\dot x(t), x(t), u(t), p) \dot{G}(t) + f_x(\dot x(t), x(t), u(t), p)G(t) \\
                &    &    & + f_p(\dot x(t), x(t), u(t), p), \\
                & \dot{F}(t)  & = & \sum_{i=1}^{n_h} w_i(t) (h^i_x(x(t)) G(t))^\top (h^i_x(x(t)) G(t)), \\
                & \dot{z}(t)  & =     & w,\\
                & x(0)        & =     & x_0, \quad  G(0) = 0, \quad F(0) = 0, \quad z(0) = 0,\\
                & x_0         & \geq  & \underbar{$x$}_0,\\
                & x_0         & \leq  & \bar{x}_0, \\
                & u(t)        & \in   & \mathcal{U},\\
                & w(t)        & \in   & \mathcal{W},\\ 
                & z(t_f) - M  & \leq  & 0,
\end{array}
$$
where $\mathcal{T} = [t_0, t_f]$ is the fixed time horizon and $x : \mathcal{T} \mapsto \mathbb{R}^{n_x}$ are the differential states. Matrix- or vector-valued inequalities and equalities are to be understood componentwise. The first and second constraint denote the dynamical system and the sensitivities of the solution of the dynamical system with respect to the uncertain parameters, respectively, and are given in an implicit form. Here, $f_{\dot x}$ ($f_x$) denotes the partial derivative of $f$ with respect to $\dot x$ ($x$). The objective $\phi(F(t_f))$ of Bolza type is a suited objective function, e.g., the A-criterion $\phi(F(t_f)) = \frac{1}{n_p} \textrm{trace}(F^{-1}(t_f))$. The evolution of the symmetric FIM $F : \mathcal{T} \mapsto \mathbb{R}^{n_p \times n_p}$ is influenced by the measurement function $h: \mathbb{R}^{n_x} \mapsto \mathbb{R}^{n_h}$, the sensitivities 
$G : \mathcal{T} \mapsto \mathbb{R}^{n_x \times n_p}$ and the sampling decisions $w(t) \in \{0,1\}^{n_h}$. The latter are the main optimization variables and represent the decision whether to measure at a given time point or not. In our direct approach, these variables are discretized, hence we write $w(t) \in \mathcal{W} := \{0,1\}^{N \times n_h}$, where $N$ is the number of discretization intervals on an equidistant time grid on $\mathcal{T}$. The sampling decisions are then accumulated in the variables $z$ and constrained by $M \in \mathbb{R}^{n_h}_{+}$. For now the controls $u \in \mathcal{U}$ are assumed to be given and fixed. The initial conditions may also be considered variables $x_0 \in \mathbb{R}^{n_x}$ with their lower and upper bounds $\underbar{x}_0$ and $\bar{x}_0$, respectively.

# Example : Lotka-Volterra Equations

The functionality in this package integrates into Julia's [`SciML`](https://sciml.ai/) ecosystem. The model is provided in symbolic form as an `ODESystem` using `ModelingToolkit.jl`[@ma2021modelingtoolkit] with additional frequency information for the observed and control variables. Both ODE and DAE system can be provided. `DynamicOED.jl` augments the given system symbolically with the its sensitivty equations and the dynamics of the FIM. The resulting system together with a sufficient information criterion defines an `OEDProblem`. Here, all sampling and control decisions are discretized in time and can be used to model additional constraints. At last, the `OEDProblem` can be transformed into an `OptimizationProblem` as a sufficient input to `Optimization.jl` [@vaibhav_kumar_dixit_2023_7738525]. Here, a variety of optimization solvers for nonlinear programming and mixed-integer nonlinear programming available as additional backends, e.g. `Ipopt` [@Waechter2006]. A simple example demonstrates the usage of `DynamicOED.jl` for the Lotka-Volterra system [@Sager2013]. 

\pagebreak

```julia
using DynamicOED
using ModelingToolkit
using Optimization, OptimizationMOI, Ipopt

@variables t
@variables x(t)=0.5 [description = "Biomass Prey"] y(t)=0.7 [description ="Biomass Predator"]
@variables u(t) [description = "Control"]
@parameters p[1:2]=[1.0; 1.0] [description = "Fixed Parameters", tunable = false]
@parameters p_est[1:2]=[1.0; 1.0] [description = "Tunable Parameters", tunable = true]
D = Differential(t)
@variables obs(t)[1:2] [description = "Observed", measurement_rate = 96]
obs = collect(obs)

# Define the system
@named lotka_volterra = ODESystem(
    [
        D(x) ~   p[1]*x - p_est[1]*x*y;
        D(y) ~  -p[2]*y + p_est[2]*x*y
    ], tspan = (0.0, 12.0),
    observed = obs .~ [x; y]
)

# Extend the system
@named oed_system = OEDSystem(lotka_volterra)

# Define a OED problem 
oed_problem = OEDProblem(structural_simplify(oed_system), DCriterion())

# Define constraints 
optimization_variables = states(oed_problem)

constraint_equations = [
    sum(optimization_variables.measurements.w₁) ≲ 32,
    sum(optimization_variables.measurements.w₂) ≲ 32,
]

@named constraint_system = ConstraintsSystem(
    constraint_equations, optimization_variables, []
)

# Solve the OED problem using Optimization.jl
optimization_problem = OptimizationProblem(
    oed_problem, AutoForwardDiff(), constraints = constraint_system,
    integer_constraints = false
)

optimal_design = solve(optimization_problem, Ipopt.Optimizer(); hessian_approximation="limited-memory")
```

\pagebreak

![State trajectory and optimal sampling design for Lotka-Volterra system. \label{fig:lotka}](figures/Lotka.pdf)


The package comes with several plotting functionalities with which the solutions can be quickly visualized and analyzed. \autoref{fig:lotka} shows the solution of the example above including the state trajectory $x(t), y(t)$ and the sampling decisions $w$.

More examples can be found at the documentation. 

# Acknowledgements

The work was funded by the German Research Foundation DFG within the priority
program 2331 'Machine Learning in Chemical Engineering' under grants KI 417/9-1, SA
2016/3-1, SE 586/25-1

# References