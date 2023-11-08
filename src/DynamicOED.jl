module DynamicOED

using DocStringExtensions

using SciMLBase
using CommonSolve
using ModelingToolkit

using ComponentArrays
#using ForwardDiff
#using ChainRulesCore
#using AbstractDifferentiation
#using Integrals

abstract type AbstractAugmentationBackened end
abstract type AbstractInformationCriterion end

"""
$(TYPEDEF)

Uses `ModelingToolkit` as a backened to augment the system.
"""
struct MTKBackend <: AbstractAugmentationBackened end

include("augment.jl")
export OEDSystem

include("fisher.jl")
export FisherIntegrand
export FisherACriterion, FisherDCriterion, FisherECriterion
export ACriterion, DCriterion, ECriterion

include("problem.jl")
export generate_objective

include("discretize.jl")
export Timegrid
#abstract type AbstractInformationCriterion end

#abstract type AbstractExperimentalDesign end
#abstract type AbstractOEDSolution end
#abstract type AbstractFisher end
#abstract type AbstractTimeGrid end
#
#build_extended_problem(::T) where T = throw(ErrorException("Augmentation for $T not implemented."))
#
#"""
#Extends the given `AbstractDEProblem` such that the dynamics include the sensitivy equations.
#"""
#function augment_problem(de::SciMLBase.AbstractDEProblem)
#    build_extended_problem(de)
#end
#
#export augment_problem
#
#include("augmentation/ode.jl")
#include("augmentation/dae.jl")
#include("augmentation/solution.jl")
#include("augmentation/utils.jl")
#include("augmentation/optimize.jl")
#include("augmentation/criteria.jl")
#include("types/timegrid.jl")
#include("types/criteria.jl")
#include("types/problem.jl")
#export OEDProblem
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
