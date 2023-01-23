"""
$(TYPEDEF)

A setup for an experimental design.

# Fields
$(FIELDS)
"""
struct ExperimentalDesign{S, P, T, PS, O} <: AbstractExperimentalDesign
    "The ODESystem"
    sys::S
    "The corresponding ODEProblem"
    prob::P
    "Indicator which parameters are used for measurements"
    w_indicator::BitVector
    "The time grid"
    tgrid::T
    "The (fixed) parameters"
    ps::PS
    "The named tuple of (extended) variables"
    variables::NamedTuple
    "The original system for a-posteriori calculation of information gain"
    sys_original::S
    "Observed variables"
    observed::O
end

function ExperimentalDesign(sys::ODESystem, time_grid = [ModelingToolkit.get_tspan(sys)]; kwargs...)
    oed_sys, F, G, z, observed = build_oed_system(sys; tspan = first(time_grid), kwargs...)
    oed_prob = ODEProblem(oed_sys)
    ps_old = parameters(sys)
    ps_new = parameters(oed_sys)
    w_idx = length(ps_old)+1:length(ps_new)
    w = BitVector(Tuple(i ∈ w_idx ? true : false for i in 1:length(ps_new)))
    p0 = Symbolics.getdefaultval.(ps_new)
    return ExperimentalDesign{typeof(oed_sys), typeof(oed_prob), typeof(time_grid), typeof(p0), typeof(observed)}(oed_sys, oed_prob, w, time_grid, p0,
        (; F = F, G = G, z = z), sys, observed
    )
end

Base.show(io::IO, oed::ExperimentalDesign) = show(io, oed.sys)
Base.summary(io::IO, oed::ExperimentalDesign) = summary(io, oed.sys)
Base.print(io::IO, oed::ExperimentalDesign) = print(io, oed.sys)


function build_oed_system(sys::ODESystem; observed = nothing, tspan = ModelingToolkit.get_tspan(sys), kwargs...)
    ## Get the eqs and the corresponding gradients
    simplified_sys = structural_simplify(sys)
    observed = isnothing(observed) ? states(simplified_sys) : observed
    eqs = map(x->x.rhs, equations(simplified_sys))
    xs = states(simplified_sys)
    ps = [p for p in parameters(simplified_sys) if istunable(p) && !isinput(p)]
    np, nx = length(ps), length(xs)
    fx = ModelingToolkit.jacobian(eqs, xs)
    fp = ModelingToolkit.jacobian(eqs, ps)
    hx = ModelingToolkit.jacobian(observed, xs)

    # Add new variables
    t = ModelingToolkit.get_iv(simplified_sys)
    D = Differential(t)
    @variables (z(t))[1:length(observed)]=zeros(length(observed)) [description="Measurement State"]
    @parameters w[1:length(observed)]=ones(length(observed)) [description="Measurement function", tunable=true]
    @variables (F(t))[1:np, 1:np]=zeros(Float64, (np,np)) [description="Fisher Information Matrix"]
    @variables (G(t))[1:nx, 1:np]=zeros(Float64, (nx,np)) [description="Sensitivity State"]

    # Build the new system of deqs
    w = collect(w)
    G = collect(G)
    hx = collect(hx)
    df = sum(enumerate(w)) do (i, wi)
        wi*((hx[i:i, :]*G)'*(hx[i:i,:]*G))
    end

    @named oed_system = ODESystem([
            equations(sys);
            vec(D.(G) .~ fx*G .+ fp);
            vec(D.(F) .~ collect(df));
            D.(z) .~ w
        ], tspan = tspan
    )
    return structural_simplify(oed_system), F, G, z, observed
end

# General predict
function (oed::ExperimentalDesign)(;u0::AbstractVector = oed.prob.u0, tspan::Tuple = ModelingToolkit.get_tspan(oed.sys), w::AbstractVector = oed.ps[oed.w_indicator], alg = Tsit5(), kwargs...)
    ps = vcat(oed.ps[.! oed.w_indicator], w)
    _prob = remake(oed.prob, u0 = u0, tspan = tspan, p = ps)
    solve(_prob, alg, kwargs...)
end

# Predict using the measurement array
function (oed::ExperimentalDesign)(w::AbstractArray; alg = Tsit5(), kwargs...)
    sol_ = nothing
    map(1:size(w, 2)) do i
        ps = vcat(oed.ps[.! oed.w_indicator], w[:,i])
        _prob = remake(oed.prob, u0 = i >= 2 ? sol_[:, end] : oed.prob.u0 , tspan = oed.tgrid[i], p = ps)
        sol_ = solve(_prob, alg, sensealg = ForwardDiffSensitivity(), kwargs...)
        return sol_
    end
end


function extract_sensitivities(oed::ExperimentalDesign, sol::AbstractArray)
    G_ = oed.variables.G
    G = vcat(map(enumerate(sol)) do (i,d)
        avoid_overlap = i == length(sol) ? 0 : 1
        d[G_][1:end-avoid_overlap]
    end...)

    hcat([g[:] for g in G]...)
end

function compute_local_information_gain(oed::ExperimentalDesign, w_::NamedTuple, kwargs...)
    w, e = w_.w, w_.regularization
    G = oed.variables.G
    sys = structural_simplify(oed.sys_original)
    xs = states(sys)

    hx = ModelingToolkit.jacobian(oed.observed, xs)
    sol = oed(w; kwargs...)
    t = [d.t[i] for d in sol for i=1:length(d)-1]
    t = [t; last(ModelingToolkit.get_tspan(oed.sys_original))]

    Pi = map(1:size(w, 1)) do i
        hi = Symbolics.value.(hx[i:i,:]) .|> Float64  # This works only for constant hx -> subsitute(...) for non-constant jacobian?
        vcat(map(1:length(sol)) do idx
            sol_i = sol[idx]
            Gi = sol_i[G]
            avoid_overlap = idx==length(sol) ? 0 : 1
            map(1:size(Gi,1)-avoid_overlap) do j
                (hi*Gi[j])'*(hi*Gi[j])
            end
        end...)
    end
    return Pi, t, sol
end

function compute_global_information_gain(oed::ExperimentalDesign, w_::NamedTuple, kwargs...)
    w, e = w_.w, w_.regularization
    F = oed.variables.F
    P, t, sol = compute_local_information_gain(oed, w_, kwargs...)
    F_ = last(last(sol)[F])
    F_inv = det(F_) > 1e-05 ? inv(F_) : nothing
    Πi = isnothing(F_inv) ? nothing : map(1:size(w, 1)) do i
        Pi = P[i]
        map(Pi) do P_i
            F_inv*P_i*F_inv
        end
    end

    return Πi, t, sol
end