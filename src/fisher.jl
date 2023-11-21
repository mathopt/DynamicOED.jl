## Criteria

function _symmetric_from_vector(x::AbstractArray{T}, ::Val{N}) where {T, N}
    return Symmetric([i <= j ? x[Int(j * (j - 1) / 2 + i)] : zero(T) for i in 1:N, j in 1:N])
end

function _symmetric_from_vector(x::AbstractArray)
    n = Int(sqrt(2 * size(x, 1) + 0.25) - 0.5)
    _symmetric_from_vector(x, Val(n))
end

function (c::C where C <: AbstractInformationCriterion)(F::AbstractArray{T, 2}, τ::R = zero(T)) where {T, R}
    c(Symmetric(F + τ * I))
end


"""
$(TYPEDEF)

The Fisher A-Criterion for experimental design. 

```julia
-tr(F)
```
"""
struct FisherACriterion <: AbstractInformationCriterion end

function (c::FisherACriterion)(F::AbstractArray{T, 2}) where T
    -tr(F)
end

"""
$(TYPEDEF)

The Fisher D-Criterion for experimental design. 

```julia
-det(F)
```
"""
struct FisherDCriterion <: AbstractInformationCriterion end

function (c::FisherDCriterion)(F::AbstractArray{T, 2}) where T
    -det(F)
end

"""
$(TYPEDEF)

The Fisher D-Criterion for experimental design. 

```julia
-min(eigvals(F))
```
"""
struct FisherECriterion <: AbstractInformationCriterion end

function (c::FisherECriterion)(F::AbstractArray{T, 2}) where T
    -minimum(real.(eigvals(F)))
end

"""
$(TYPEDEF)

The A-Criterion for experimental design. 

```julia
tr(inv(F))
```
"""
struct ACriterion <: AbstractInformationCriterion end

function (c::ACriterion)(F::AbstractArray{T, 2}) where T
    λ = inv.(eigvals(F))
    sum(real.(λ))
end


"""
$(TYPEDEF)

The D-Criterion for experimental design. 

```julia
det(inv(F))
```
"""
struct DCriterion <: AbstractInformationCriterion end

function (c::DCriterion)(F::AbstractArray{T, 2}) where T
    inv(det(F))
end


"""
$(TYPEDEF)

The E-Criterion for experimental design. 

```julia
max(eigvals(F))
```
"""
struct ECriterion <: AbstractInformationCriterion end

function (c::ECriterion)(F::AbstractArray{T, 2}) where T
    λ = eigvals(F) 
    maximum(real.(λ))
end

