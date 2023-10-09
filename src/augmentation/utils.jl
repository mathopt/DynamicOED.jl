struct TimeGrid{T,Ts,I} <: AbstractTimeGrid
    "Minimal time grid"
    simgrid::T
    "Individual time grids for sampling and control"
    grids::Ts
    "Indicator which class and variable does change"
    indicator::I
end

"""
$(SIGNATURES)

Constructs a `TimeGrid` from a given timespan and numbers of discretization intervals
(equidistant) for sampling decisions `Nw` and controls `Nc`, respectively.

The resulting (equidistant) timegrids for sampling decisions and controls are merged into a common
`simgrid`. Indicator vectors representing which indices of sampling decisions and controls
are to be used on which section of the `simgrid` are saved in `indicator`.
"""
function TimeGrid(tspan::Tuple{Real, Real}, Nw::Int, Nc::Int; tol = 1e-6)
    wgrid = DynamicOED.get_tgrid(tspan, Nw)
    cgrid = DynamicOED.get_tgrid(tspan, Nc)

    completegrid = sort(vcat(first.(wgrid),first.(cgrid), last.(wgrid), last.(cgrid)))
    completegrid = completegrid[vcat(true, diff(completegrid) .> tol)]

    simgrid = Tuple.([(completegrid[i], completegrid[1+i]) for i=1:length(completegrid)-1])

    w_indicator = zeros(Int, length(simgrid))
    c_indicator = zeros(Int, length(simgrid))

    for (i, subinterval) in enumerate(simgrid)
        for (j, w_interval) in enumerate(wgrid)
            if (first(subinterval) >= first(w_interval) && last(subinterval) <= last(w_interval))
                w_indicator[i] = j
            end
        end
    end

    for (i, subinterval) in enumerate(simgrid)
        for (j, c_interval) in enumerate(cgrid)
            if (first(subinterval) >= first(c_interval) && last(subinterval) <= last(c_interval))
                c_indicator[i] = j
            end
        end
    end

    grids       = (wgrid=wgrid, cgrid=cgrid)
    indicators  = (w=w_indicator, c=c_indicator)

    return TimeGrid{typeof(simgrid),typeof(grids),typeof(indicators)}(simgrid, grids, indicators)
end

function get_tgrid(tspan::Tuple{Real, Real}, Δt::AbstractFloat)
    @assert Δt < -(reverse(tspan)...) "Stepsize must be smaller than total time interval."
    first_ts = first(tspan):Δt:(last(tspan)-Δt)
    last_ts = (first(tspan)+Δt):Δt:last(tspan)
    tgrid = collect(zip(first_ts, last_ts))
    if !isapprox(last(last(tgrid)),last(tspan))
        push!(tgrid, (last(last(tgrid)), last(tspan)))
    end
    tgrid
end

function get_tgrid(tspan::Tuple{Real, Real}, N::Int)
    Δt = -(reverse(tspan)...)/N
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
    sols = grid_solve(x.predictor, vcat(res.u0, x.predictor.problem.u0[nx+1:end]), x.predictor.problem.p, tuple(x.timegrid.simgrid...); kwargs...)
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