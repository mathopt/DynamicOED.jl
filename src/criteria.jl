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
   -minimum(abs.(eigvals(last(sol[F]))))
end

struct ACriterion <: AbstractInformationCriterion end

function (c::ACriterion)(ed::ExperimentalDesign, w::AbstractArray, e; kwargs...)
    F = ed.variables.F
    sol = last(ed(w; kwargs...))
    F_ = last(sol[F])
    #(F_[1,1] + F_[2,2]) / (F_[1,1]*F_[2,2]-F_[1,2]*F_[2,1])
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
    maximum(abs.(eigvals(F_)))
end

# TODO'S:

# Initialisierung? Inequality constraint -> erster Aufruf von inv(F) fails weil F = 0
    # ✓ erstmal behoben mit eps auf Hauptdiagonale und erneutem Ändern der lower bounds
# Regularisierung -> eps als Variable? evtl. penalisieren oder gegen Null treiben?
    # ✓ done
# SolutionObject definen -> InformationGain, SolutionTrajectory, LagrangeMultiplier (dispatchen auf Solver)
    # ✓ done
# PlotRecipe für SOlution Object
    # wenn Lagrange da plotte die mit, sonst nicht
# FAKTOR 2 in Information Gain? Wo kommt der her?

# Wenn Zeit ist:
# Funktion reginverse(A, eps) -> berechnet inverse mit regularisierung
# Funtkion traceinverse(A) -> berechnet trace
# Funktion frule_traceinverse(A) -> Formel(37) aus Sager2012
# FrankWolfe.jl? Boscia.jl?