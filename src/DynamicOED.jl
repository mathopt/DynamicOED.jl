module DynamicOED

using DocStringExtensions
using LinearAlgebra

using AbstractDifferentiation
using ForwardDiff

using ModelingToolkit
using Symbolics

using SciMLBase
using OrdinaryDiffEq
using SciMLSensitivity

using Nonconvex

abstract type AbstractExperimentalDesign end
abstract type AbstractInformationCriterion end

include("experimental_design.jl")
export ExperimentalDesign

include("criteria.jl")
export FischerACriterion, FischerDCriterion, FischerECriterion

include("optimize.jl")
export solve

end # module DynamicOED
