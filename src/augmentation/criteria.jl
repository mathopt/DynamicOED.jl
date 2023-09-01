struct DCriterion <: AbstractInformationCriterion end

function (c::DCriterion)(F::AbstractArray{T, 2}) where T
    inv(det(F))
end

function apply_criterion(c::AbstractInformationCriterion,
                        ed::OEDProblem, w::AbstractArray; kwargs...)

    F = ed(w; kwargs...)
    c(F)
end

function apply_criterion(c::AbstractInformationCriterion,
                        ed::OEDProblem, w::W; kwargs...) where W <: NamedTuple


    u0_ = vcat(w.u0, ed.predictor.problem.u0[size(w.u0,1)+1:end])
    F = ed(w.w, u0_; kwargs...)
    c(F)
end