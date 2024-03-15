function generate_timegrid(x::Num, tspan::Tuple)
    @assert is_measured(x) "Provided variable $x has no rate. No time grid can be created!"
    Δt = get_measurement_rate(x)
    _generate_timegrid(Δt, tspan)
end

function _generate_timegrid(Δt::T, tspan::Tuple{Real, Real}) where {T <: Real}
    @assert Δt>0 "Stepsize must be greater than 0."
    @assert Δt<-(reverse(tspan)...) "Stepsize must be smaller than total time interval."
    t0, tinf = tspan
    timepoints = collect(T, t0:Δt:tinf)
    if timepoints[end] != tinf
        push!(timepoints, tinf)
    end
    return timepoints
end

function _generate_timegrid(N::Int, tspan::Tuple{Real, Real})
    Δt = -(reverse(tspan)...) / N
    _generate_timegrid(Δt, tspan)
end

"""
$(TYPEDEF)

A structure for holding a multi-variable time grid. 

# Fields

$(FIELDS)
"""
struct Timegrid{V, I, G, T}
    "The variables"
    variables::V
    "The indicator for switching variables"
    indicators::I
    "The individual time grids"
    timegrids::G
    "The overall time grid"
    timespans::T
end

function Timegrid(tspan::Tuple, x::Num...; time_tolerance::Real = 1e-6)
    @assert all(is_measured, x) "Not all variables have rates associated with them. Can not generate time grid."

    _timegrids = map(x) do xi
        generate_timegrid(xi, tspan)
    end

    _completegrid = reduce(vcat, _timegrids)
    sort!(_completegrid)
    unique!(_completegrid)

    completegrid = [tspan_
                    for tspan_ in zip(_completegrid[1:(end - 1)], _completegrid[2:end])
                    if abs(-(tspan_...)) >= time_tolerance]

    timegrids = map(_timegrids) do grid
        [tspan_
         for tspan_ in zip(grid[1:(end - 1)], grid[2:end])
         if abs(-(tspan_...)) >= time_tolerance]
    end

    indicators = zeros(Int, length(x), size(completegrid, 1))
    @inbounds for i in axes(indicators, 1), j in axes(indicators, 2),
        k in axes(timegrids[i], 1)

        t_start, t_stop = completegrid[j]
        t_min, t_max = timegrids[i][k]
        if t_start >= t_min && t_stop <= t_max
            indicators[i, j] = k
        end
    end

    return Timegrid(Symbol.(x), indicators, timegrids, completegrid)
end

function Timegrid(sys::ModelingToolkit.AbstractODESystem,
        tspan = ModelingToolkit.get_tspan(sys))
    c = get_control_parameters(sys)
    w = get_measurement_function(sys)
    Timegrid(tspan, c..., w...)
end

@inline get_tspan(grid::Timegrid, i::Int) = getindex(grid.timespans, i)
@inline _get_variable_idx(grid::Timegrid, var::Symbol) = findfirst(==(var), grid.variables)

function get_variable_idx(grid::Timegrid, var::Symbol, i::Int)
    id = _get_variable_idx(grid, var)
    isnothing(id) && return 1 # We always assume here that this will work 
    return grid.indicators[id, i]
end

function get_vars_from_grid(grid::Timegrid, i::Int, nt::NamedTuple{names}) where {names}
    map(names) do name
        idx = get_variable_idx(grid, name, i)
        getindex(getfield(nt, name), idx)
    end
end

# We have three categories of tuneables in and OEDProblem
# States
# This includes the fisher states !
# ( And possibly shooting variables )
# Parameters
# Initial conditions -> Need to be merged into u0
# Controls -> Need to be merged with the parameter defaults
# Measurements -> Need to be merged with the parameter defaults

function generate_initial_variables(sys::ModelingToolkit.AbstractODESystem, tgrid::Timegrid)
    ics = get_initial_conditions(sys)
    controls = get_control_parameters(sys)
    measurements = get_measurement_function(sys)

    initial_conditions = NamedTuple(map(ics) do ic
        (Symbol(ic), Symbolics.getdefaultval(ic))
    end)

    control_variables = NamedTuple(map(controls) do control
        c_sym = Symbol(control)
        idx = _get_variable_idx(tgrid, c_sym)
        (c_sym, [Symbolics.getdefaultval(control) for _ in axes(tgrid.timegrids[idx], 1)])
    end)

    measurement_variables = NamedTuple(map(measurements) do w
        w_sym = Symbol(w)
        idx = _get_variable_idx(tgrid, w_sym)
        (w_sym, [Symbolics.getdefaultval(w) for _ in axes(tgrid.timegrids[idx], 1)])
    end)

    regularization = 1.0

    (;
        initial_conditions, controls = control_variables,
        measurements = measurement_variables,
        regularization = regularization) |> sortkeys |> ComponentVector
end

function generate_variable_bounds(sys::ModelingToolkit.AbstractODESystem,
        tgrid::Timegrid,
        lower = false)
    ics = get_initial_conditions(sys)
    controls = get_control_parameters(sys)
    measurements = get_measurement_function(sys)

    initial_conditions = NamedTuple(map(ics) do ic
        bounds_ = getbounds(ic)
        bound = lower ? first(bounds_) : last(bounds_)
        (Symbol(ic), bound)
    end)

    control_variables = NamedTuple(map(controls) do control
        c_sym = Symbol(control)
        idx = _get_variable_idx(tgrid, c_sym)
        bounds_ = getbounds(control)
        bound = lower ? first(bounds_) : last(bounds_)
        (c_sym, [bound for _ in axes(tgrid.timegrids[idx], 1)])
    end)

    measurement_variables = NamedTuple(map(measurements) do w
        w_sym = Symbol(w)
        idx = _get_variable_idx(tgrid, w_sym)
        bounds_ = getbounds(w)
        bound = lower ? first(bounds_) : last(bounds_)
        (w_sym, [bound for _ in axes(tgrid.timegrids[idx], 1)])
    end)

    regularization = lower ? eps() : 1.0
    (;
        initial_conditions, controls = control_variables,
        measurements = measurement_variables,
        regularization = regularization) |> sortkeys |> ComponentVector
end

struct ParameterRemake <: Function
    "The control indices inside the parameter vector"
    control_idx::Vector{Int}
    "The measurement indices inside the parameter vector"
    measure_idx::Vector{Int}
    "The initial condition indices inside the parameter vector"
    ic_param_idx::Vector{Int}

    function ParameterRemake(sys::ModelingToolkit.AbstractODESystem)
        params = Dict([si => i for (i, si) in enumerate(parameters(sys))])
        controls = get_control_parameters(sys)
        measurements = get_measurement_function(sys)
        ics = get_initial_conditions(sys)
        control_map = ModelingToolkit.varmap_to_vars(params, controls, tofloat = false)
        measure_map = ModelingToolkit.varmap_to_vars(params, measurements, tofloat = false)
        ic_map = ModelingToolkit.varmap_to_vars(params, ics, tofloat = false)
        control_map = isnothing(control_map) ? Int[] : control_map
        measure_map = isnothing(measure_map) ? Int[] : measure_map
        ic_map = isnothing(ic_map) ? Int[] : ic_map

        new(control_map, measure_map, ic_map)
    end
end

function __find_and_return(i, idx, x0, xrpl)
    isempty(idx) && return x0
    idx = findfirst(==(i), idx)
    isnothing(idx) ? x0 : xrpl[idx]
end

function (remake::ParameterRemake)(p0::AbstractVector, measurements, controls, ics)
    map(eachindex(p0)) do i
        val = __find_and_return(i, remake.measure_idx, p0[i], measurements)
        val = __find_and_return(i, remake.control_idx, val, controls)
        __find_and_return(i, remake.ic_param_idx, val, ics)
    end
end

struct StateRemake <: Function
    ic_state_idx::Vector{Int}

    function StateRemake(sys::ModelingToolkit.AbstractODESystem)
        ic_state_idx = get_initial_condition_id.(get_initial_conditions(sys))
        ic_state_idx = isnothing(ic_state_idx) ? Int[] : ic_state_idx
        new(ic_state_idx)
    end
end

function (remake::StateRemake)(u0::AbstractVector, ics)
    map(eachindex(u0)) do i
        __find_and_return(i, remake.ic_state_idx, u0[i], ics)
    end
end

struct OEDRemake
    grid::Timegrid
    parameter_remake::ParameterRemake
    state_remake::StateRemake
    p_prototype::ComponentVector

    function OEDRemake(sys::ModelingToolkit.AbstractODESystem,
            tspan = ModelingToolkit.get_tspan(sys),
            grid = Timegrid(sys, tspan))
        parameter_remake = ParameterRemake(sys)
        state_remake = StateRemake(sys)
        return new(grid,
            parameter_remake,
            state_remake,
            generate_initial_variables(sys, grid) .* 0.0)
    end
end

function (remaker::OEDRemake)(i::Int,
        prob::P,
        parameters::ComponentVector{T},
        u0::AbstractVector = prob.u0,
        p0::AbstractVector = prob.p) where {P, T}
    ics = getproperty(parameters, :initial_conditions) |> NamedTuple
    controls = getproperty(parameters, :controls) |> NamedTuple
    measurements = getproperty(parameters, :measurements) |> NamedTuple
    # Get the right controls 
    controls = get_vars_from_grid(remaker.grid, i, controls)
    measurements = get_vars_from_grid(remaker.grid, i, measurements)
    p0_ = remaker.parameter_remake(p0, measurements, controls, ics)
    tspan = get_tspan(remaker.grid, i)
    if i == 1
        u0_ = remaker.state_remake(u0, ics)
        return remake(prob, u0 = u0_, p = p0_, tspan = tspan)
    end
    return remake(prob, u0 = u0, p = p0_, tspan = tspan)
end

# TODO Adjust for better conversion of the parameter vector
function sequential_solve(remaker::OEDRemake,
        prob::P,
        alg::A,
        parameters::AbstractVector{T};
        kwargs...) where {P, A, T}
    sequential_solve(remaker, prob, alg, parameters + remaker.p_prototype; kwargs...)
end

function sequential_solve(remaker::OEDRemake,
        prob::P,
        alg::A,
        parameters::ComponentVector{T};
        kwargs...) where {P, A, T}
    u0 = T.(copy(prob.u0))
    p0 = T.(copy(prob.p))
    _sequential_solve(remaker,
        prob,
        alg,
        parameters,
        u0,
        p0,
        tuple(axes(remaker.grid.timespans, 1)...);
        kwargs...)
end

function _sequential_solve(remaker::OEDRemake,
        prob::P,
        alg::A,
        parameters::ComponentVector{T},
        u0::AbstractVector{T},
        p0::AbstractVector{T},
        idxs::TP;
        kwargs...)::Tuple{
        AbstractArray{T, 2}, AbstractVector{T}} where {P, A, T, TP <: Tuple}
    __sequential_solve(remaker, prob, alg, parameters, u0, p0, idxs...)
end

function __sequential_solve(remaker::OEDRemake,
        prob::P,
        alg::A,
        parameters::ComponentVector{T},
        u0::AbstractVector{T},
        p0::AbstractVector{T},
        idxs::Vararg{Int};
        kwargs...) where {P, A, T}
    id = first(idxs)
    sol = solve(remaker(id, prob, parameters, u0, p0),
        alg;
        saveat = remaker.grid.timespans[id],
        kwargs...)
    x_i = Array(sol)
    t_i = sol.t
    x_j, t_j = __sequential_solve(remaker,
        prob,
        alg,
        parameters,
        sol[:, end],
        p0,
        Base.tail(idxs)...;
        kwargs...)
    return (hcat(x_i, x_j[:, 2:end]), vcat(t_i, t_j[2:end]))
end

function __sequential_solve(remaker::OEDRemake,
        prob::P,
        alg::A,
        parameters::ComponentVector{T},
        u0::AbstractVector{T},
        p0::AbstractVector{T},
        idxs::Int;
        kwargs...) where {P, A, T}
    sol = solve(remaker(idxs, prob, parameters, u0, p0),
        alg;
        saveat = remaker.grid.timespans[idxs],
        kwargs...)
    Array(sol), sol.t
end
