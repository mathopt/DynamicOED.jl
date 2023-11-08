

struct TimeGrid{T,Ts,I}
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
function TimeGrid(tspan::Tuple{Real, Real}, Nw::Int, Nc::Int = Nc; tol = 1e-6)
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
