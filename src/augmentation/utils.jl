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

function Base.show(io::IO, sol::OEDSolution)
    println("Exited with final objective $(sol.obj).")
end

function get_t_and_sols(x::DynamicOED.OEDProblem{true}, res::NamedTuple; kwargs...)
    nx = x.predictor.dimensions.nx
    sols = grid_solve(x.predictor, vcat(res.u0, x.predictor.problem.u0[nx+1:end]), x.predictor.problem.p, tuple(x.timegrid...); kwargs...)
    solt = vcat([sol.t[1:end-1] for sol in sols]..., last(last(sols).t))
    syms = reshape(sols[1].prob.f.syms .|> string, (1,length(first(sols[1]))))
    sols_ = hcat([Array(sol[:,1:end-1]) for sol in sols]..., last(last(sols)))
    return (t=solt, u=sols_, syms=syms)
end

function get_t_and_sols(x::DynamicOED.OEDProblem{false}, args...; kwargs...)
    solt = vcat([sol.t[1:end-1] for sol in x.sols]..., last(last(x.sols).t))
    syms = reshape(x.sols[1].prob.f.syms .|> string, (1,length(first(x.sols[1]))))
    sols_ = hcat([Array(sol[:,1:end-1]) for sol in x.sols]..., last(last(x.sols)))
    return (t=solt, u=sols_, syms=syms)
end

function get_lagrange_multiplier(res)
    try
        return res.problem.mult_g
    catch e
        return nothing
    end
end

function compute_local_information_gain(oed::OEDProblem, sol::AbstractArray, t::AbstractVector)
    nh  = oed.predictor.dimensions.nh
    nx  = oed.predictor.dimensions.nx
    np  = oed.predictor.dimensions.np
    p   = oed.predictor.problem.p
    map(1:nh) do i
        map(zip(eachcol(sol), t)) do (sol_t, t_i)
            x, G = sol_t[1:nx], reshape(sol_t[nx+1:end], (nx, np))
            hxi = oed.predictor.observed.hx(x, p, t_i)[i:i, :]
            (hxi*G)' * (hxi*G)
        end
    end
end

function compute_global_information_gain(oed::OEDProblem, F_tf::AbstractArray,
                                        local_information_gain::AbstractArray)

    isapprox(det(F_tf), 0) && return nothing
    F_inv = inv(F_tf)

    map(1:oed.predictor.dimensions.nh) do i
        map(local_information_gain[i]) do Pi
            F_inv * Pi * F_inv
        end
    end
end