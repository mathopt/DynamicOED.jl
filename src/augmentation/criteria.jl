struct DCriterion <: AbstractInformationCriterion end

function (c::DCriterion)(F::AbstractArray{T, 2}) where T
    inv(det(F))
end

function apply_criterion(c::AbstractInformationCriterion,
                        ed::OEDProblem, w::AbstractArray; kwargs...)

    F = ed(w; kwargs...)
    #F = _symmetric_from_vector(F_)
    c(F)
end