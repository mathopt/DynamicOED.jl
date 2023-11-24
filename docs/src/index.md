# DynamicOED.jl

Repository for optimal experimental design for differential equations using optimal control.

`DynamicOED.jl` uses multiple packages of Julia's [SciML](https://sciml.ai/) ecosystem, especially [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl), [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) and [Optimization.jl](https://github.com/SciML/Optimization.jl) to define [optimal experimental design problems using optimal control](https://doi.org/10.1137/110835098).

## Installation 

**This package is not registered yet, but once it is you can use this section.**

Assuming that you already have Julia correctly installed, it suffices to import
DynamicOED.jl in the standard way:

```julia
import Pkg
Pkg.add("DynamicOED")
```

The packages relevant to the core functionality of DynamicOED.jl will be imported
accordingly and, in most cases, you do not have to worry about the manual
installation of dependencies. However, you will need to add the specific optimizer
packages.

To solve the underlying optimization problem, please refer to [Optimization.jl](https://github.com/SciML/Optimization.jl) for available solvers.

## Features

+ Currently we support Ordinary Differential Equations and Differential Algebraic Equations.
+ Relaxed and Integer formulations of the underlying problem
+ Unknown initial conditions
+ Continuous and discrete controls (in terms of the variable)
+ Variable (measurement) rates for observed and control variables
+ Custom constraints 

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
    sum(optimization_variables.measurements.w₁) ≲ 3,
]

@named constraint_set = ConstraintsSystem(constraints, optimization_variables,[])

# Initialize the optimization problem
optimization_problem = OptimizationProblem(oed_problem, AutoForwardDiff(),
      constraints = constraint_set,
      integer_constraints = false)

# Solven for the optimal values of the observed variables
solve(optimization_problem, optimizer)
``` 

