function _symmetric_from_vector(x::AbstractArray{T}) where T
    # Find number n such that n*(n+1)/2 = length(x)
    n = Int(sqrt(2 * length(x) + 0.25) - 0.5)
    _symmetric_from_vector(x, n)
end

"""
$(SIGNATURES)

Converts AbstractArray of size `n*(n+1)/2` into a `Symmetric` matrix of size `(n,n)`.
"""
function _symmetric_from_vector(x::AbstractArray{T}, n::Int) where T
    return Symmetric([ i<=j ? x[Int(j*(j-1)/2+i)] : zero(T) for i=1:n, j=1:n])
end

"""
$(SIGNATURES)

Integrates dynamics for current iterate `x` and evaluates criterion `c` on Fisher information matrix.
"""
function apply_criterion(c, ed::ExperimentalDesign, x; kwargs...)
    w = x.w
    τ = x.τ
    F_ = ed.variables.F
    sol = last(ed(w; kwargs...))
    F = _symmetric_from_vector(last(sol[F_]))
    c(F, τ)
end

struct FisherACriterion <: AbstractInformationCriterion end

function (c::FisherACriterion)(F::AbstractArray{T, 2}, τ::T) where T
   -tr(F)
end

struct FisherDCriterion <: AbstractInformationCriterion end

function (c::FisherDCriterion)(F::AbstractArray{T, 2}, τ::T) where T
   -det(F)
end

struct FisherECriterion <: AbstractInformationCriterion end

function (c::FisherECriterion)(F::AbstractArray{T, 2}, τ::T) where T
   -minimum(abs.(eigvals(F)))
end

struct ACriterion <: AbstractInformationCriterion end

function (c::ACriterion)(F::AbstractArray{T, 2}, τ::T) where T
    tr(inv(F+τ*I))
end

struct DCriterion <: AbstractInformationCriterion end

function (c::DCriterion)(F::AbstractArray{T, 2}, τ::T) where T
    det(inv(F+τ*I))
end

struct ECriterion <: AbstractInformationCriterion end

function (c::ECriterion)(F::AbstractArray{T, 2}, τ::T) where T
    maximum(abs.(eigvals(inv(F+τ*I))))
end

"""
$(SIGNATURES)

Evaluates the so-called switching function for the FisherACriterion.
This quantity is needed for the sufficient condition for `w*(t) = w_max`, where
`w*(t)` is the optimal sampling decision in the `OEDSolution` and `w_max` is the upper bound on `w(t)`
"""
function switching_function(res::OEDSolution{FisherACriterion})
    np  = sum(res.oed.w_indicator)
    sw = [tr.(P)/np for P in res.information_gain.local_information_gain]
    return (sw, "trace P(t)")
end

"""
$(SIGNATURES)

Evaluates the so-called switching function for the FisherDCriterion.
This quantity is needed for the sufficient condition for `w*(t) = w_max`, where
`w*(t)` is the optimal sampling decision in the `OEDSolution` and `w_max` is the upper bound on `w(t)`
"""
function switching_function(res::OEDSolution{FisherDCriterion})
    F_ = res.oed.variables.F
    np  = sum(res.oed.w_indicator)
    F = _symmetric_from_vector(last(last(res.sol)[F_]))
    C = inv(F)
    detF = det(F) / np
    sw = map(res.information_gain.local_information_gain) do P
        detF .* [sum(C .* Pᵢ) for Pᵢ in P]
    end
    return (sw,  "det F(tf) ⋅ ∑ C(tf) ∘ P(t)")
end

"""
$(SIGNATURES)

Evaluates the so-called switching function for the FisherECriterion.
This quantity is needed for the sufficient condition for `w*(t) = w_max`, where
`w*(t)` is the optimal sampling decision in the `OEDSolution` and `w_max` is the upper bound on `w(t)`
"""
function switching_function(res::OEDSolution{FisherECriterion})
    F_ = res.oed.variables.F
    F = _symmetric_from_vector(last(last(res.sol)[F_]))
    eigenF = eigen(F)
    λ_min, idx_min = findmin(sign.(eigenF.values) .* abs.(eigenF.values))
    v = eigenF.vectors[:,idx_min:idx_min]
    sw = map(enumerate(res.information_gain.local_information_gain)) do (i,P)
        [(v' * Pᵢ * v)[1,1] for Pᵢ in P]
    end
    return (sw, "v^T P(t) v")
end

"""
$(SIGNATURES)

Evaluates the so-called switching function for the ACriterion.
This quantity is needed for the sufficient condition for `w*(t) = w_max`, where
`w*(t)` is the optimal sampling decision in the `OEDSolution` and `w_max` is the upper bound on `w(t)`
"""
function switching_function(res::OEDSolution{ACriterion})
    np  = sum(res.oed.w_indicator)
    sw = [tr.(C)/np for C in res.information_gain.global_information_gain]
    return (sw, "trace Π(t)")
end

"""
$(SIGNATURES)

Evaluates the so-called switching function for the DCriterion.
This quantity is needed for the sufficient condition for `w*(t) = w_max`, where
`w*(t)` is the optimal sampling decision in the `OEDSolution` and `w_max` is the upper bound on `w(t)`
"""
function switching_function(res::OEDSolution{DCriterion})
    F_ = res.oed.variables.F
    np  = sum(res.oed.w_indicator)
    F = _symmetric_from_vector(last(last(res.sol)[F_]))
    detC = det(inv(F)) / np
    sw = map(res.information_gain.global_information_gain) do Π
        detC .* [sum(F .* Πᵢ) for Πᵢ in Π]
    end
    return (sw,  "det C(tf) ⋅ ∑ F(tf) ∘ Π(t)")
end

"""
$(SIGNATURES)

Evaluates the so-called switching function for the ECriterion.
This quantity is needed for the sufficient condition for `w*(t) = w_max`, where
`w*(t)` is the optimal sampling decision in the `OEDSolution` and `w_max` is the upper bound on `w(t)`
"""
function switching_function(res::OEDSolution{ECriterion})
    F_ = res.oed.variables.F
    F = _symmetric_from_vector(last(last(res.sol)[F_]))
    eigenC = eigen(inv(F))
    λ_max, idx_max = findmax(sign.(eigenC.values) .* abs.(eigenC.values))
    v = eigenC.vectors[:,idx_max:idx_max]
    sw = map(enumerate(res.information_gain.global_information_gain)) do (i,Π)
        [(v' * Πᵢ * v)[1,1] for Πᵢ in Π]
    end
    return (sw, "v^T Π(t) v")
end

function _supported_criteria()
    return [FisherACriterion(), FisherDCriterion(), FisherECriterion(), ACriterion(), DCriterion(), ECriterion()]
end