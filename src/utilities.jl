function continuous_predict(oed::OEDProblem, result::ComponentVector)
    # Build the remaker and the problem 
    timegrid = oed.timegrid
    tspan = (first(first(timegrid.timespans)), last(last(timegrid.timespans)))
    odae_prob = ODAEProblem(oed.system, Pair[], tspan)
    remaker = OEDRemake(oed.system, tspan, timegrid)
    # Right now map, switch to pmap if available
    u0 = copy(odae_prob.u0)
    p0 = copy(odae_prob.p)
    
    map(axes(timegrid.timespans, 1)) do i 
        prob_ = remaker(i, odae_prob, result, u0, p0)
        sol = solve(prob_, oed.alg, oed.diffeq_options...)
        u0 .= sol[:, end]
        @info prob_.tspan
        sol
    end
end

function extract_Qs(oed::OEDProblem, result::ComponentVector{T}, sols = continuous_predict(oed, result)) where T
    # We extract the unweighted sensitivities here
    observed_eqs = observed(oed.system)
    m = size(filter(!is_measurement_function, filter(istunable, parameters(oed.system))), 1)
    qs = filter(is_information_gain, reduce(vcat, Symbolics.get_variables.(observed_eqs)))
    n = round(Int, size(qs,1) / m)
    Qs = Matrix{T}[]
    @inbounds for i in eachindex(sols)
        q = map(xi->reshape(xi, n, m), sols[i][qs])
        q = i == 1 ? q : q[2:end]
        foreach(eachindex(q)) do j
                push!(Qs, q[j])
        end
    end
    Qs
end

function extract_fisher(oed::OEDProblem, result::ComponentVector{T}, sols = continuous_predict(oed, result)) where T
    fs = filter(is_fisher_state, states(oed.system))
    n = Val(Int(sqrt(2 * size(fs, 1) + 0.25) - 0.5))
    f_vec = last(last(sols)[fs])
    _symmetric_from_vector(f_vec, n)
end

function compute_local_information_gain(Qs::AbstractVector)
    n_vars = size(first(Qs), 1)
    map(Qs) do Qi
        (Qi'Qi) / n_vars
    end
end

function compute_information_gain(oed::OEDProblem, result::ComponentVector{T}) where T
    sols = continuous_predict(oed, result)
    Qs = extract_Qs(oed, result, sols)
    F = extract_fisher(oed, result, sols)
    Finv = inv(F)
    Ps = compute_local_information_gain(Qs)
    Πs = map(eachindex(Ps)) do i 
        Finv*Ps[i]*Finv
    end
    # np is simply the sum over all measurements
    np = sum(map(is_measurement_function, parameters(oed.system)))
    (F, Ps, Πs, np, sols)
end