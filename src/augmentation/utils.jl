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

function Base.show(io::IO, oed::OEDProblem)
    printstyled(io, "OEDProblem"; color=:green)
    println(io, " with $(oed.predictor.dimensions.np) parameters \
                and $(oed.predictor.dimensions.nh) observation functions.")
    print(io, "Contains augmented ")
    show(io, "text/plain", oed.predictor.problem)
end

function get_t_and_sols(x::DynamicOED.OEDProblem{true}, res::NamedTuple)
    nx = x.predictor.dimensions.nx
    sols = grid_solve(x.predictor, vcat(res.u0, x.predictor.problem.u0[nx+1:end]), x.predictor.problem.p, tuple(x.timegrid...))
    solt = vcat([sol.t for sol in sols]...)
    syms = reshape(sols[1].prob.f.syms .|> string, (1,length(first(sols[1]))))
    return (t=solt, sol=hcat(Array.(sols)...), syms=syms)
end

function get_t_and_sols(x::DynamicOED.OEDProblem{false}, args...)
    solt = vcat([sol.t for sol in x.sols]...)
    syms = reshape(x.sols[1].prob.f.syms .|> string, (1,length(first(x.sols[1]))))
    return (t=solt, sol=hcat(Array.(x.sols)...), syms=syms)
end