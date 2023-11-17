# DynamicOED.jl

[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) [![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle) ![Lifecycle:Experimental](https://img.shields.io/badge/Lifecycle-Experimental-339999)

Repository for optimal experimental design for differential equations using optimal control.

`DynamicOED.jl` extends multiple packages of Julia's [SciML](https://sciml.ai/) ecosystem, especially [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl), [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) and [Optimization.jl](https://github.com/SciML/Optimization.jl). 

Currently we support both Ordinary Differential Equations and Differential Algebraic Equations. 

## Example

```julia
using DynamicOED
using ModelingToolkit
using Optimization, OptimizationMOI, Ipopt

# Define the differential equations
@variables t
@variables x(t)=1.0 [description = "State"]
@parameters p[1:1]=-2.0 [description = "Fixed parameter", tunable = true]
@variables obs(t) [description = "Observed", measurement_rate = 10]
D = Differential(t)

@named simple_system = ODESystem([
        D(x) ~ p[1] * x,
    ], tspan = (0.0, 1.0),
    observed = obs .~ [x.^2])

@named oed = OEDSystem(simple_system)
oed = structural_simplify(oed)

# Augment the original problem to an OED problem
oed_problem = OEDProblem(structural_simplify(oed), crit)

# Define an MTK Constraint system over the grid variables
optimization_variables = states(oed_problem)
        
constraints = [
    0 ≲ 0.2 .- 0.5 * sum(Δts.w₁ .* optimization_variable.measurements.w₁),
]

@named constraint_set = ConstraintsSystem(constraints, optimization_variables,[])

# Initialize the optimization problem
optimization_problem = OptimizationProblem(oed_problem, AutoForwardDiff(),
      constraints = constraint_set,
      integer_constraints = false)

# Solven for the optimal values of the observed variables
solve(opt_prob, optimizer)
``` 

