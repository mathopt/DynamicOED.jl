function _predict_F(::C where C <: AbstractInformationCriterion, ed::ExperimentalDesign, w::AbstractArray; kwargs...)
    F = ed.variables.F
    sol = last(ed(w; kwargs...))
    last(sol[F])
end

struct FischerACriterion <: AbstractInformationCriterion end

function (c::FischerACriterion)(ed::ExperimentalDesign, w::AbstractArray; kwargs...)
    F = ed.variables.F
    sol = last(ed(w; kwargs...))
   -tr(last(sol[F]))
end

struct FischerDCriterion <: AbstractInformationCriterion end

function (c::FischerDCriterion)(ed::ExperimentalDesign, w::AbstractArray; kwargs...)
    F = ed.variables.F
    sol = last(ed(w; kwargs...))
   -det(last(sol[F]))
end

## TODO: SOMEHOW MAKE EIGVALS COMPATIBLE WITH AD/ZYGOTE
## e.g., https://discourse.julialang.org/t/native-eigenvals-for-differentiable-programming/27126
struct FischerECriterion <: AbstractInformationCriterion end

function (c::FischerECriterion)(ed::ExperimentalDesign, w::AbstractArray; kwargs...)
    F = ed.variables.F
    sol = last(ed(w; kwargs...))
   -minimum(abs.(eigvals(last(sol[F]))))
end

struct ACriterion <: AbstractInformationCriterion end