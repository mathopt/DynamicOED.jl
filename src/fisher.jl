### Fisher Matrix

# Helper struct to compute the integral of the FIM
struct FisherIntegrand{F <: Function, T <: Function, A <: SciMLBase.AbstractIntegralAlgorithm}
    "Reduced dynamics of the fisher information matrix "
    observed::F
    "The transformation of the reduced fisher information matrix to the full (symmetric) form"
    transform::T
    "Quadrature algorithm"
    alg::A 

    function FisherIntegrand(f::F, transform::T, alg::A = QuadGKJL(); kwargs...) where {F <: Function, T, A <: SciMLBase.AbstractIntegralAlgorithm}
        new{F, T, A}(f, transform, alg)
    end
end

function _symmetric_from_vector(x::AbstractArray{T}, n::Int) where T
    return Symmetric([ i<=j ? x[Int(j*(j-1)/2+i)] : zero(T) for i=1:n, j=1:n])
end

function FisherIntegrand(sys::ModelingToolkit.ODESystem; integral_algorithm = QuadGKJL(), kwargs...)
    # Given that we have a very specific problem structure, we can simply 
    # build a new equation here
    target_eqs = [eq for eq in observed(sys) if any(is_fisher_state, Symbolics.get_variables(eq))]
    
    np =Int(sqrt(2 * size(target_eqs, 1) + 0.25) - 0.5) # Size of the fisher

    transformation = let np = np
        (x) -> _symmetric_from_vector(x, np)
    end

    f = build_function(map(x->x.rhs, target_eqs), states(sys), parameters(sys), ModelingToolkit.get_iv(sys), expression = Val{false}) 
   
    FisherIntegrand(first(f), transformation, integral_algorithm; kwargs...)
end

function __integrate(f::FisherIntegrand, x::AbstractArray, t::AbstractVector, p::AbstractArray)
    y = map(axes(x, 2)) do i 
        f.observed(x[:, i],  p, t[i])
    end
    problem = Integrals.SampledIntegralProblem(y, t)
    res = solve(problem, TrapezoidalRule())
    Array(res)
end

function (f::FisherIntegrand)(x, t, p::AbstractArray)
    __integrate(f, x, t, p)
end

function (f::FisherIntegrand)(prob::SciMLBase.AbstractDEProblem, u0::AbstractArray, p::AbstractArray, tspan::Tuple{T, T}, alg::SciMLBase.AbstractDEAlgorithm; kwargs...) where T
    _prob = remake(prob, u0 = u0, p = p, tspan = tspan)
    sol = solve(_prob, alg; kwargs...)
    __integrate(f, sol, p)
end

(f::FisherIntegrand)(x::AbstractVector) = f.transform(x)


# Custom rules to only use ForwardDiff here
# IDK why, but Zygote crashes.
# Credit to mohamed82008 
# https://github.com/JuliaNonconvex/NonconvexUtils.jl/blob/282c272a47042b5ffa6a5665c9803227cc0a3269/src/abstractdiff.jl#L5
#function ChainRulesCore.rrule(x::FisherIntegrand, sol::SciMLBase.DESolution, p::AbstractArray)
#    v, (∇,) = AbstractDifferentiation.value_and_jacobian(AbstractDifferentiation.ForwardDiffBackend(), Base.Fix1(x, sol), p)
#    return v, Δ -> (NoTangent(), NoTangent(), ∇'*Δ)
#end

#function ChainRulesCore.frule(
#    (_, Δx), x::FisherIntegrand,  sol::SciMLBase.DESolution, p::AbstractArray
#)
#    v, (∇,) = AbstractDifferentiation.value_and_jacobian(AbstractDifferentiation.ForwardDiffBackend(), Base.Fix1(x, sol), p)
#
#    return v, ∇ * Δx
#end

#function ChainRulesCore.rrule(x::FisherIntegrand, prob::SciMLBase.AbstractDEProblem, u0::AbstractArray, p::AbstractArray, tspan::Tuple{T, T}, alg::SciMLBase.AbstractDEAlgorithm; kwargs...) where T
#    v, (∇u,∇p,) = AbstractDifferentiation.value_and_jacobian(AbstractDifferentiation.ForwardDiffBackend(), (u, p) -> x(prob, u, p, tspan, alg; kwargs...), u0, p)
#    return v, Δ -> begin 
#        F = x.transform(Δ)
#        F_xp = hcat(F, I(size(∇p, 2)-size(F, 1)))
#        @info ∇u 
#        @info F
#        (NoTangent(), NoTangent(), NoTangent(), ∇u'*F, NoTangent(), NoTangent(), NoTangent())
#    end
#end

#function ChainRulesCore.frule(
#    (_, Δx), x::FisherIntegrand, prob::SciMLBase.AbstractDEProblem, u0::AbstractArray, p::AbstractArray, tspan::Tuple{T, T}, alg::SciMLBase.AbstractDEAlgorithm; kwargs...) where T
#)
#    v, (∇u,∇p,) = AbstractDifferentiation.value_and_jacobian(AbstractDifferentiation.ForwardDiffBackend(), (u, p) -> x(prob, u, p, tspan, alg; kwargs...), u0, p)
#
#    return v, (∇u)
#end


## Criteria

"""
$(TYPEDEF)

The Fisher A-Criterion for experimental design. 

```julia
-tr(F)
```
"""
struct FisherACriterion <: AbstractInformationCriterion end

function (c::FisherACriterion)(F::AbstractArray{T, 2}, τ::T = zero(T)) where T
   -tr(F + τ*I)
end



"""
$(TYPEDEF)

The Fisher D-Criterion for experimental design. 

```julia
-det(F)
```
"""
struct FisherDCriterion <: AbstractInformationCriterion end

function (c::FisherDCriterion)(F::AbstractArray{T, 2}, τ::T = zero(T)) where T
   -det(F + τ*I)
end


"""
$(TYPEDEF)

The Fisher D-Criterion for experimental design. 

```julia
-min(abs, eigvals(F))
```
"""
struct FisherECriterion <: AbstractInformationCriterion end

function (c::FisherECriterion)(F::AbstractArray{T, 2}, τ::T = zero(T)) where T
   -minimum(abs.(eigvals(F + τ*I)))
end


"""
$(TYPEDEF)

The A-Criterion for experimental design. 

```julia
tr(inv(F + τ * I))
```

where `τ` is a small regularization constant.
"""
struct ACriterion <: AbstractInformationCriterion end

function (c::ACriterion)(F::AbstractArray{T, 2}, τ::T = zero(T)) where T
    tr(inv(F+τ*I))
end

"""
$(TYPEDEF)

The D-Criterion for experimental design. 

```julia
det(inv(F + τ * I))
```

where `τ` is a small regularization constant.
"""
struct DCriterion <: AbstractInformationCriterion end

function (c::DCriterion)(F::AbstractArray{T, 2}, τ::T = zero(T)) where T
    inv(det(F+τ*I))
end



"""
$(TYPEDEF)

The E-Criterion for experimental design. 

```julia
max(abs, eigvals(inv(F + τ * I)))
```

where `τ` is a small regularization constant.
"""
struct ECriterion <: AbstractInformationCriterion end

function (c::ECriterion)(F::AbstractArray{T, 2}, τ::T = zero(T)) where T
    maximum(abs.(eigvals(inv(F+τ*I))))
end
