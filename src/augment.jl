
"""
$(TYPEDEF)

Indicator that a given variable is a measurement function. 

```
@variables w=1.0 [measurement_function=true]
```
"""
struct MeasurementFunction end

Symbolics.option_to_metadata_type(::Val{:measurement_function}) = MeasurementFunction

is_measurement_function(x::Num) = is_measurement_function(Symbolics.unwrap(x))
is_measurement_function(x) = Symbolics.getmetadata(x, MeasurementFunction, false)
set_measurement_function(x) = Symbolics.setmetadata(x, MeasurementFunction, true)

"""
$(TYPEDEF)

Indicator that a given variable is a state of the fisher information matrix. 

```
@variables F[1:3, 1:3] [fisher_state=true]
```
"""
struct FisherState end

Symbolics.option_to_metadata_type(::Val{:fisher_state}) = FisherState

is_fisher_state(x::Num) = is_fisher_state(Symbolics.unwrap(x))
is_fisher_state(x) = Symbolics.getmetadata(x, FisherState, false)
set_fisher_state(x) = Symbolics.setmetadata(x, FisherState, true)

struct UnweightedInformationGain end

function Symbolics.option_to_metadata_type(::Val{:unweighted_information_gain})
    UnweightedInformationGain
end

is_information_gain(x::Num) = is_information_gain(Symbolics.unwrap(x))
is_information_gain(x) = Symbolics.getmetadata(x, UnweightedInformationGain, false)

"""
$(TYPEDEF)

Indicator that a given state variable's initial condition is unknown.

```
@variables x(t)=1.0 [variable_ic=true]
```
"""
struct VariableIC end

Symbolics.option_to_metadata_type(::Val{:variable_ic}) = VariableIC

is_initial_condition(x::Num) = is_initial_condition(Symbolics.unwrap(x))
is_initial_condition(x) = hasmetadata(x, VariableIC) # Symbolics.getmetadata(x, VariableIC, false)
set_initial_condition(x, i::Int) = Symbolics.setmetadata(x, VariableIC, i)
get_initial_condition_id(x::Num) = get_initial_condition_id(Symbolics.unwrap(x))
get_initial_condition_id(x) = Symbolics.getmetadata(x, VariableIC, 0)

"""
$(TYPEDEF)

Indicator that a given state is subject to a fixed rate.
Is used for modeling the rate of observation of observed variables and the rate of control for control variables.
If the provided rate is a `Real`, it is assumed that the resulting time grid is given in fractions of the time unit. 
If the provided rate is a `Int`, it is assumed that the resulting time grid is divided in `rate` equidistant intervals. 

```
@variables y(t) [measurement_rate=0.1] # Create a variable measured every 0.1 t
@variables y(t) [measurement_rate=0.1] # Create a variable measured every 0.1 t
@parameters c [input=true, measurement_rate=2] # Create a control variable which acts 2 times over the course of the simulation
```
"""
struct VariableRate end

Symbolics.option_to_metadata_type(::Val{:measurement_rate}) = VariableRate

is_measured(x::Num) = is_measured(Symbolics.unwrap(x))
is_measured(x) = hasmetadata(x, VariableRate) # Symbolics.getmetadata(x, VariableIC, false)

set_measurement_rate(x, dt) = begin
    @assert dt>0 "Measurement rate must be greater than zero!"
    ret = Symbolics.setmetadata(x, VariableRate, dt)
    ret
end

set_measurement_rate(x::Num, dt) = begin
    set_measurement_rate(Symbolics.unwrap(x), dt) |> Num
end

get_measurement_rate(x::Num) = get_measurement_rate(Symbolics.unwrap(x))
get_measurement_rate(x) = Symbolics.getmetadata(x, VariableRate, 0.0)

## Helper functions for constraint generation
function get_measurement_function(sys::ModelingToolkit.AbstractSystem)
    filter(is_measurement_function, parameters(sys)) |> Base.Fix1(collect, Num)
end
function get_initial_conditions(sys::ModelingToolkit.AbstractSystem)
    filter(is_initial_condition, parameters(sys)) |> Base.Fix1(collect, Num)
end
function get_control_parameters(sys::ModelingToolkit.AbstractSystem)
    filter(ModelingToolkit.isinput, parameters(sys)) |> Base.Fix1(collect, Num)
end
function get_fisher_states(sys::ModelingToolkit.AbstractSystem)
    filter(is_fisher_state, states(sys)) |> Base.Fix1(collect, Num)
end
##

function construct_jacobians(::MTKBackend,
        sys::ModelingToolkit.AbstractODESystem,
        p = parameters(sys))
    eqs = map(x -> x.rhs - x.lhs, equations(sys))
    t = ModelingToolkit.get_iv(sys)
    D = Differential(t)
    fx = ModelingToolkit.jacobian(eqs, states(sys))
    dfddx = ModelingToolkit.jacobian(eqs, D.(states(sys)))
    fp = ModelingToolkit.jacobian(eqs, p)
    obs = filter(x -> is_measured(x.lhs), observed(sys))
    obs = isempty(obs) ? states(sys) : map(x -> x.rhs, obs)
    hx = ModelingToolkit.jacobian(obs, states(sys))
    return dfddx, fx, fp, hx
end

function build_augmented_system(sys::ModelingToolkit.AbstractODESystem,
        backend::AbstractAugmentationBackened;
        name::Symbol,
        kwargs...)
    T = Float64
    # The set of tuneable parameters
    p = parameters(sys)
    # The set of controls 
    c = filter(ModelingToolkit.isinput, p)
    # The set of tuneable parameters
    p_tuneable = setdiff(filter(ModelingToolkit.istunable, p), c)
    # The states
    x = states(sys)
    # The unknown initial conditions
    x_ic = eltype(x)[]
    unknown_initial_conditions = Int[]
    @inbounds for i in axes(x, 1)
        xi = getindex(x, i)
        if istunable(xi)
            xi_0 = Symbolics.variable(Symbol(xi, "₀"), T = Symbolics.symtype(xi))
            xi_0 = setmetadata(xi_0,
                ModelingToolkit.VariableDescription,
                "Initial condition of state $(xi)")
            xi_0 = ModelingToolkit.setdefault(xi_0, ModelingToolkit.getdefault(xi))
            xi_0 = ModelingToolkit.toparam(xi_0)
            xi_0 = setmetadata(xi_0, ModelingToolkit.VariableTunable, true)
            xi_0 = setmetadata(xi_0, ModelingToolkit.VariableBounds, getbounds(xi))
            xi_0 = set_initial_condition(xi_0, i)
            push!(unknown_initial_conditions, i)
            push!(x_ic, Symbolics.unwrap(xi_0))
            push!(p_tuneable, Symbolics.unwrap(xi_0))
        end
    end

    # The independent variable 
    t = ModelingToolkit.get_iv(sys)
    delta_t = Differential(t)
    # The observed equations 
    obs = filter(x -> is_measured(x.lhs), observed(sys))
    @info obs
    # Check if all observed equations and controls have measurement rates associated
    @assert !isempty(obs) "None of the observed equations have measurement rates associated to them! Please provide at least one observable measurement."
    @assert all(is_measured, c) "Not all controls have rates associated to them! If you mean to apply continuous controls, please adjust your model before passing it."

    @assert !isempty(obs) "Please defined `observed` equations to use optimal experimental design."

    np, nx, n_obs = length(p_tuneable), length(x), length(obs)
    ## build the jacobians
    dfx, fx, fp, hx = construct_jacobians(backend, sys, p_tuneable)

    # Check the size of the equations
    @assert size(fx, 1)==size(fp, 1)==size(x, 1) "The size of the state equations and the jacobian does not match"
    @assert size(fp, 2)==size(p_tuneable, 1) "The size of the state equations and the jacobian does not match"

    G_init = zeros(T, (nx, np))

    if !isempty(x_ic)
        n_ic = size(x_ic, 1)
        G_init[unknown_initial_conditions, np .- (unknown_initial_conditions .- 1)] .= one(T) .*
                                                                                       I(n_ic)
    end

    ## Define new variables
    # TODO Maybe switch to variables here.
    @variables F(t)[1:np, 1:np]=zeros(T, (np, np)) [
        description = "Fisher Information Matrix",
        fisher_state = true,
    ]
    @variables G(t)[1:nx, 1:np]=G_init [description = "Sensitivity State"]
    @variables Q(t)[1:n_obs, 1:np] [
        description = "Unweighted Fisher Information Derivative",
        unweighted_information_gain = true,
    ]

    w = Vector{Num}(undef, n_obs)
    @inbounds for i in axes(w, 1)
        wi = Symbolics.variable(:w, i, T = Int)
        obsi = obs[i].lhs
        wi = setmetadata(wi,
            ModelingToolkit.VariableDescription,
            "Measurement function of $obsi")
        wi = ModelingToolkit.setdefault(wi, 0.0)
        wi = set_measurement_rate(wi, get_measurement_rate(obsi))
        wi = setmetadata(wi, ModelingToolkit.VariableTunable, true)
        wi = setmetadata(wi, MeasurementFunction, true)
        wi = setmetadata(wi, ModelingToolkit.VariableBounds, (0, 1))
        wi = ModelingToolkit.toparam(wi)
        w[i] = Num(wi)
    end
    # Build the new system of deqs
    F = collect(F)
    G = collect(G)
    Q = collect(Q)

    # Create new observed function 
    idx = triu(trues(np, np))
    new_obs = delta_t.(F[idx]) .~ (sum(enumerate(w)) do (i, wi)
        wi * ((hx[i:i, :] * G)' * (hx[i:i, :] * G))[idx]
    end)

    dynamic_eqs = equations(sys)
    # We always assume DAE form here. Results in a stable system
    # 60 % of the time it works everytime! This is an easteregg, do not take it seriously!
    sens_eqs = vec(zeros(T, nx, np) .~ dfx * delta_t.(G) .+ fx * G .+ fp)

    # We do not need to do anything more than push this into the equations
    # Simplify will figure out the rest
    ODESystem([vec(dynamic_eqs);
            vec(sens_eqs);
            vec(new_obs);
            vec(Q .~ hx * G)],
        t,
        vcat(x, vec(G), vec(F[idx]), vec(Q)),
        vcat(union(p, p_tuneable), w),
        tspan = ModelingToolkit.get_tspan(sys),
        observed = observed(sys), name = name)
end

function OEDSystem(sys::ModelingToolkit.AbstractODESystem; kwargs...)
    build_augmented_system(sys, MTKBackend(); kwargs...)
end
