module DynamicOED

using DocStringExtensions
using LinearAlgebra

using FastDifferentiation
using AbstractDifferentiation
#using ForwardDiff
using Integrals
using ChainRulesCore
#using ModelingToolkit
#using ArrayInterface
#using Symbolics

using SciMLBase
using OrdinaryDiffEq
using SciMLSensitivity
using Zygote
using Nonconvex
using NonconvexIpopt
#using CairoMakie
#using Reexport

#@reexport using CairoMakie: save

abstract type AbstractExperimentalDesign end
abstract type AbstractInformationCriterion end
abstract type AbstractOEDSolution end
abstract type AbstractFisher end

build_extended_problem(::T) where T = throw(ErrorException("Augmentation for $T not implemented."))

"""
Extends the given `AbstractDEProblem` such that the dynamics include the sensitivy equations.
"""
function augment_problem(de::SciMLBase.AbstractDEProblem)
    build_extended_problem(de)
end

export augment_problem

include("augmentation/ode.jl")
include("augmentation/dae.jl")
include("augmentation/solution.jl")
include("augmentation/utils.jl")
include("augmentation/optimize.jl")
include("augmentation/criteria.jl")

#include("experimental_design/experimental_design.jl")
#include("experimental_design/ode.jl")
#include("experimental_design/dae.jl")
#export ExperimentalDesign
#
#include("optimize.jl")
#export OEDSolution, solve
#
#include("criteria.jl")
#export FisherACriterion, FisherDCriterion, FisherECriterion
#export ACriterion, DCriterion, ECriterion
#
#include("plotting.jl")
#export plotOED

end # module DynamicOED
