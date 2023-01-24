function _predict_F(::C where C <: AbstractInformationCriterion, ed::ExperimentalDesign, w::AbstractArray; kwargs...)
    F = ed.variables.F
    sol = last(ed(w; kwargs...))
    last(sol[F])
end

struct FisherACriterion <: AbstractInformationCriterion end

function (c::FisherACriterion)(ed::ExperimentalDesign, w::AbstractArray, e; kwargs...)
    F = ed.variables.F
    sol = last(ed(w; kwargs...))
   -tr(last(sol[F]))
end

struct FisherDCriterion <: AbstractInformationCriterion end

function (c::FisherDCriterion)(ed::ExperimentalDesign, w::AbstractArray, e; kwargs...)
    F = ed.variables.F
    sol = last(ed(w; kwargs...))
   -det(last(sol[F]))
end

## TODO: SOMEHOW MAKE EIGVALS COMPATIBLE WITH AD/ZYGOTE
## e.g., https://discourse.julialang.org/t/native-eigenvals-for-differentiable-programming/27126
struct FisherECriterion <: AbstractInformationCriterion end

function (c::FisherECriterion)(ed::ExperimentalDesign, w::AbstractArray, e; kwargs...)
    F = ed.variables.F
    sol = last(ed(w; kwargs...))
   -minimum(abs.(eigvals(Symmetric(last(sol[F])))))
end

struct ACriterion <: AbstractInformationCriterion end

function (c::ACriterion)(ed::ExperimentalDesign, w::AbstractArray, e; kwargs...)
    F = ed.variables.F
    sol = last(ed(w; kwargs...))
    F_ = last(sol[F])
    tr(inv(F_+e*I))
end

struct DCriterion <: AbstractInformationCriterion end

function (c::DCriterion)(ed::ExperimentalDesign, w::AbstractArray, e; kwargs...)
    F = ed.variables.F
    sol = last(ed(w; kwargs...))
    F_ = last(sol[F])
    det(inv(F_+e*I))
end

struct ECriterion <: AbstractInformationCriterion end
# TODO: Same as for FisherECriterion
function (c::ECriterion)(ed::ExperimentalDesign, w::AbstractArray, e; kwargs...)
    F = ed.variables.F
    sol = last(ed(w; kwargs...))
    F_ = last(sol[F])
    maximum(abs.(eigvals(inv(Symmetric(F_)))))
end

function switching_function(res::OEDSolution{FisherACriterion})
    np  = sum(res.oed.w_indicator)
    sw = [tr.(P)/np for P in res.information_gain.local_information_gain]
    return (sw, "trace P(t)")
end

function switching_function(res::OEDSolution{FisherDCriterion})
    F_ = res.oed.variables.F
    np  = sum(res.oed.w_indicator)
    F = last(last(res.sol)[F_])
    C = inv(F)
    detF = det(F) / np
    sw = map(res.information_gain.local_information_gain) do P
        detF .* [sum(C .* Pᵢ) for Pᵢ in P]
    end
    return (sw,  "det F(tf) ⋅ ∑ C(tf) ∘ P(t)")
end

function switching_function(res::OEDSolution{FisherECriterion})
    F_ = res.oed.variables.F
    F = last(last(res.sol)[F_])
    eigenF = eigen(F)
    λ_min, idx_min = findmin(sign.(eigenF.values) .* abs.(eigenF.values))
    v = eigenF.vectors[:,idx_min:idx_min]
    sw = map(enumerate(res.information_gain.local_information_gain)) do (i,P)
        [(v' * Pᵢ * v)[1,1] for Pᵢ in P]
    end
    return (sw, "v^T P(t) v")
end


function switching_function(res::OEDSolution{ACriterion})
    np  = sum(res.oed.w_indicator)
    sw = [tr.(C)/np for C in res.information_gain.global_information_gain]
    return (sw, "trace Π(t)")
end

function switching_function(res::OEDSolution{DCriterion})
    F_ = res.oed.variables.F
    np  = sum(res.oed.w_indicator)
    F = last(last(res.sol)[F_])
    detC = det(inv(F)) / np
    sw = map(res.information_gain.global_information_gain) do Π
        detC .* [sum(F .* Πᵢ) for Πᵢ in Π]
    end
    return (sw,  "det C(tf) ⋅ ∑ F(tf) ∘ Π(t)")
end

function switching_function(res::OEDSolution{ECriterion})
    F_ = res.oed.variables.F
    F = last(last(res.sol)[F_])
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

# TODO'S:

# Initialisierung? Inequality constraint -> erster Aufruf von inv(F) fails weil F = 0
    # ✓ erstmal behoben mit eps auf Hauptdiagonale und erneutem Ändern der lower bounds
# Regularisierung -> eps als Variable? evtl. penalisieren oder gegen Null treiben?
    # ✓ done
# SolutionObject definen -> InformationGain, SolutionTrajectory, LagrangeMultiplier (dispatchen auf Solver)
    # ✓ done
# PlotRecipe für SOlution Object
    # Plotte solution
    # Plotte sampling decision
    # Plotte Lagrange Multiplier wenn verfügbar
    # Plotte Sensitivities
# FAKTOR 2 in Information Gain? Wo kommt der her?

# Wenn Zeit ist:
# Funktion reginverse(A, eps) -> berechnet inverse mit regularisierung
# Funtkion traceinverse(A) -> berechnet trace
# Funktion frule_traceinverse(A) -> Formel(37) aus Sager2012
# FrankWolfe.jl? Boscia.jl?