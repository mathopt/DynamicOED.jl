## Criteria

function _symmetric_from_vector(x::AbstractArray{T}, ::Val{N}) where {T, N}
    return Symmetric([ i<=j ? x[Int(j*(j-1)/2+i)] : zero(T) for i=1:N, j=1:N])
end

function _symmetric_from_vector(x::AbstractArray)
    n = Int(sqrt(2 * size(x, 1) + 0.25) - 0.5)
    _symmetric_from_vector(x, Val(n))
end


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
