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

