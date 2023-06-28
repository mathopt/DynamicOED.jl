module DynamicOED

using DocStringExtensions
using LinearAlgebra

using AbstractDifferentiation
using ForwardDiff

using ModelingToolkit
using ArrayInterface
using Symbolics

using SciMLBase
using OrdinaryDiffEq
using SciMLSensitivity

using Nonconvex
using NonconvexIpopt
using CairoMakie
using Reexport

@reexport using CairoMakie: save

abstract type AbstractExperimentalDesign end
abstract type AbstractInformationCriterion end
abstract type AbstractOEDSolution end

include("experimental_design.jl")
export ExperimentalDesign

include("optimize.jl")
export OEDSolution, solve

include("criteria.jl")
export FisherACriterion, FisherDCriterion, FisherECriterion
export ACriterion, DCriterion, ECriterion

include("plotting.jl")
export plotOED

include("dae.jl")
export modelingtoolkitize

end # module DynamicOED
