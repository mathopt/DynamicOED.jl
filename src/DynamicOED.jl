module DynamicOED

using DocStringExtensions

using SciMLBase
using CommonSolve
using ModelingToolkit
using LinearAlgebra

using ComponentArrays
using OrdinaryDiffEq
using SciMLSensitivity
using Optimization

abstract type AbstractAugmentationBackened end
abstract type AbstractInformationCriterion end

# Credit to https://discourse.julialang.org/t/sort-keys-of-namedtuple/94630/3
@generated sortkeys(nt::NamedTuple{KS}) where {KS} = :(NamedTuple{
    $(Tuple(sort(collect(KS)))),
}(nt))

"""
$(TYPEDEF)

Uses `ModelingToolkit` as a backened to augment the system.
"""
struct MTKBackend <: AbstractAugmentationBackened end

include("augment.jl")
export OEDSystem

include("fisher.jl")
export FisherACriterion, FisherDCriterion, FisherECriterion
export ACriterion, DCriterion, ECriterion

include("discretize.jl")

include("problem.jl")
export OEDProblem
export get_timegrids
export get_initial_variables

include("utilities.jl")

end # module DynamicOED
