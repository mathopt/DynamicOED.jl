## Criteria

function _symmetric_from_vector(x::AbstractArray{T}, ::Val{N}) where {T, N}
    return Symmetric([ i<=j ? x[Int(j*(j-1)/2+i)] : zero(T) for i=1:N, j=1:N])
end

function _symmetric_from_vector(x::AbstractArray)
    n = Int(sqrt(2 * size(x, 1) + 0.25) - 0.5)
    _symmetric_from_vector(x, Val(n))
end


"""
$(SIGNATURES)

Returns the switching function of the corresponding criterion. 
"""
function get_switching_function(x) 
    @error "The current criterion $x does not have any switching function associated!"
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


function get_switching_function(::FisherACriterion)
    (Fs, Ps, Πs, np) -> (tr.(Ps) ./ np, Symbol("tr(P(t))"))
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


function get_switching_function(::FisherDCriterion)
    (F, Ps, Πs, np) -> begin 
        Finv = inv(F)
        detF = det(F) / np
        f_val = map(Ps) do Pi 
            detF .* sum(Finv .* Pi)
        end
        (f_val, Symbol("det(F(∞))(∑ C(∞) ⊙ P(t))"))
    end
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

function get_switching_function(::FisherECriterion)
    (F, Ps, Πs, np) -> begin 
        eigF = eigen(F)
        λ_min, id_min = findmin(eigF.values)
        v = eigF.vectors[:, id_min:id_min]
        f_val = map(Ps) do Pi 
            only(v'*Pi*v)
        end
        (f_val, Symbol("vᵀP(t)v"))
    end
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

function get_switching_function(::ACriterion)
    (F, Ps, Πs, np) -> begin 
        (tr.(Πs) ./ np, Symbol("tr(Π(t))"))
    end
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

function get_switching_function(::DCriterion)
    (F, Ps, Πs, np) -> begin 
        detC = inv(det(F)) / np
        f_val = map(Πs) do Π 
            detC .* sum(F .* Π)
        end
        (f_val, Symbol("det(C(∞))(∑ F(∞) ⊙ Π(t))"))
    end
end

# For ForwardDiff
function (c::AbstractInformationCriterion)(F::AbstractArray{T, 2}, τ::R) where {T, R}
    c(F, T(τ))
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

function get_switching_function(::ECriterion)
    # We use the fact that λ(inv(F)) = inv(λ)(F)  for regular matrices
    # the eigenvectors stay the same.

    (F, Ps, Πs, np) -> begin 
        eigF = eigen(F) 
        id_min = argmin(eigF.values)
        v = eigF.vectors[:, id_min:id_min]
        f_val = map(Πs) do Π 
            v'*Π*v
        end
        (f_val, Symbol("vᵀΠ(t)v"))
    end
end


