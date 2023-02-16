function plotOED(res::OEDSolution; idxs=states(res.oed.sys_original), f = Figure())
    tspan       = (first(res.sol).t[1], last(res.sol).t[end])
    Δt          = -(reverse(tspan)...)/length(res.sol)
    tControl    = first(tspan):Δt:last(tspan)
    n_vars      = sum(res.oed.w_indicator)
    t           = res.information_gain.t
    gain        = [tr.(gain_)/n_vars for gain_ in res.information_gain.global_information_gain]

    plot_multiplier = !isnothing(res.multiplier) && res.criterion in _supported_criteria()

    # Plot states
    t   = vcat([s.t[1:end-1] for s in res.sol]...)
    t   = [t; [last(res.sol).t[end]]]
    sol = vcat([s[idxs][1:end-1] for s in res.sol]...)
    sol = hcat([sol; [last(res.sol)[idxs][end]]]...)
    labels = collect(idxs .|> string)

    ax0 = Axis(f[1,1:n_vars], title="Differential states", xlabel="Time")
    for (i, row) in enumerate(eachrow(sol))
        lines!(ax0, t, row, label=labels[i])
    end
    f[1, n_vars+1] = Legend(f, ax0)

    # Plot sensitivities
    labelsG = collect(vec(res.oed.variables.G)) .|> string .|> remove_excess_parentheses_and_whitespace
    ax21 = Axis(f[2,1:n_vars], title="Sensitivities", xlabel="Time")
    foreach(enumerate(eachrow(res.information_gain.sensitivities))) do (i,g)
        lines!(ax21, t, g, label=labelsG[i])
    end
    f[2, n_vars+1] = Legend(f, ax21, nbanks=size(res.oed.variables.G,1))

    # Plot conditions for measuring or not depending on the used criterion
    H_w =  plot_multiplier ?  switching_function(res) : (gain, "trace Π(t)")
    label_criterion = last(H_w)
    axs = []
    for (i, g_) in enumerate(first(H_w))
        ax1 = Axis(f[3,i], title="Sampling w$i(t)", xlabel="Time")
        ax2 = Axis(f[3,i], yticklabelcolor=:red, yaxisposition=:right)

        p1 = stairs!(ax1, tControl, [res.w.w[i,1]; res.w.w[i,:]], color=:black, stairs=:pre, label="w(t)")
        p2 = lines!(ax2, t, g_, color=:red, label=label_criterion)
        if plot_multiplier
            p3 = hlines!(ax2, res.multiplier[i], color=:red, linestyle=:dash, label="μ")
        end

        if i == n_vars
            ps = plot_multiplier ? [p1, p2, p3] : [p1, p2]
            ls = plot_multiplier ? ["w(t)", label_criterion, "μ"] : ["w(t)", label_criterion]
            f[3, n_vars+1] = Legend(f, ps, ls)
        end
        push!(axs, ax2)
        ax2.yaxisposition       = :right
        ax2.yticklabelalign     = (:left, :center)
        ax2.xticklabelsvisible  = false
        ax2.ygridstyle          = :dash
        ax2.ygridcolor          = :grey
        ax2.xticklabelsvisible  = false
        ax2.xlabelvisible       = false
    end
    linkyaxes!(axs...)

    f
end

function remove_excess_parentheses_and_whitespace(str::String)
    idx1 = first(findfirst("(", str))
    idx1 == first(findlast("(", str)) && return str # only remove outermost set of parentheses
    idx2 = first(findlast(")", str))

    str = str[[i for i=1:length(str) if i != idx1 && i != idx2]]
    filter(x -> !isspace(x), str)
end
