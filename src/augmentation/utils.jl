function get_tgrid(Δt::AbstractFloat, tspan::Tuple{Real, Real})
    @assert Δt < -(reverse(tspan)...) "Stepsize must be smaller than total time interval."
    first_ts = first(tspan):Δt:(last(tspan)-Δt)
    last_ts = (first(tspan)+Δt):Δt:last(tspan)
    tgrid = collect(zip(first_ts, last_ts))
    if !isapprox(last(last(tgrid)),last(tspan))
        push!(tgrid, (last(last(tgrid)), last(tspan)))
    end
    tgrid
end

function _symmetric_from_vector(x::AbstractArray{T}, n::Int) where T
    return Symmetric([ i<=j ? x[Int(j*(j-1)/2+i)] : zero(T) for i=1:n, j=1:n])
end

function _symmetric_from_vector(x::AbstractArray{T}) where T
    # Find number n such that n*(n+1)/2 = length(x)
    n = Int(sqrt(2 * length(x) + 0.25) - 0.5)
    _symmetric_from_vector(x, n)
end