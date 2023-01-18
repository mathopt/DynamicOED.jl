#@recipe function plot(res::OEDSolution, idxs=states(res.oed.sys_original))
#    n_plots     = 2
#    n_meas      = length(res.information_gain.global_information_gain)
#    tspan       = (first(res.sol).t[1], last(res.sol).t[end])
#    tControl    = first(tspan):(last(tspan)-first(tspan))/length(res.sol):last(tspan)
#
#    layout  := (n_plots + n_meas)
#    grid    := true
#    size    := (900,900)
#
#    for (i, d) in enumerate(res.sol)
#        labels = i == 1 ? reshape(idxs, 1, length(idxs)) : nothing
#        @series begin
#            seriestype  := :path
#            xlims       := tspan
#            labels      := labels
#            title       := "Differential states"
#            subplot     := 1
#            color       := [i for i=1:size(idxs,1)]'
#            idxs        := idxs
#            d
#        end
#    end
#
#    @series begin
#        seriestype  := :path
#        subplot     := 2
#        title       := "Sensitivities"
#        labels      := hcat(["G$i(t)" for i in axes(res.information_gain.sensitivities, 1)]...)
#        res.information_gain.t, res.information_gain.sensitivities'
#    end
#
#    for (i, gain) in enumerate(res.information_gain.global_information_gain)
#
#        @series begin
#            seriestype  := :path
#            link        := :x
#            subplot     := 2 + i
#            label       := "trace Π$i(t)"
#            res.information_gain.t, tr.(gain)
#        end
#        # TODO: FIND OUT HOW TO PLOT IN SAME SUBPLOT WITH TWO Y AXIS, see, e.g. twinx() in Plots.jl
#        @series begin
#            seriestype  := :steppre
#            subplot     := 2+i
#            label       := "w$i(t)"
#            tControl, [res.w.w[i,1]; res.w.w[i,:]]
#        end
#
#        @series begin
#            seriestype  := :hline
#            linestyle   := :dash
#            subplot     := 2 + i
#            title       := "Information gain Π$i \nand sampling w$i"
#            label       := "μ$i"
#            [res.multiplier[i]]
#        end
#    end
#end

function plotOED(res::OEDSolution; idxs=states(res.oed.sys_original), f = Figure())
    tspan       = (first(res.sol).t[1], last(res.sol).t[end])
    Δt = -(reverse(tspan)...)/length(res.sol)
    tControl = first(tspan):Δt:last(tspan)

    t    = res.information_gain.t
    gain = [tr.(gain_) for gain_ in res.information_gain.global_information_gain]
    G    = res.information_gain.sensitivities

    t   = vcat([s.t[1:end-1] for s in res.sol]...)
    t   = [t; [last(res.sol).t[end]]]
    sol = vcat([s[idxs][1:end-1] for s in res.sol]...)
    sol = hcat([sol; [last(res.sol)[idxs][end]]]...)
    labels = idxs .|> string

    nCols = length(gain)
    plot_multiplier = !isnothing(res.multiplier)

    ax0 = Axis(f[1,1:nCols], title="Differential states", xlabel="Time")
    for (i, row) in enumerate(eachrow(sol))
        lines!(ax0, t, row, label=labels[i])
    end
    f[1, nCols+1] = Legend(f, ax0)

    ax21 = Axis(f[2,1:nCols], title="Sensitivities", xlabel="Time")
    for (i, g) in enumerate(eachrow(G))
        lines!(ax21, t, g, label="G$i(t)")
    end
    nbanks_ = size(G,1) > 1 ? size(G,1) ÷ 4 : 1
    f[2, nCols+1] = Legend(f, ax21, nbanks=nbanks_)



    axs = []
    for (i,g_) in enumerate(gain)
        ax1 = Axis(f[3,i], title="Sampling w$i(t)", xlabel="Time")
        ax2 = Axis(f[3,i], yticklabelcolor=:red, yaxisposition=:right)

        p1 = stairs!(ax1, tControl, [res.w.w[i,1]; res.w.w[i,:]], color=:black, stairs=:pre, label="w$i(t)")
        p2 = lines!(ax2, t, g_, color=:red, label="Π$i(t)")
        if plot_multiplier
            p3 = hlines!(ax2, res.multiplier[i], color=:red, linestyle=:dash, label="μ$i")
        end

        if i == nCols
            ps = plot_multiplier ? [p1, p2, p3] : [p1, p2]
            ls = plot_multiplier ? ["w(t)", "trace Π(t)", "μ"] : ["w(t)", "trace Π(t)"]
            leg = Legend(f[3,nCols+1], ps, ls)
        end
        push!(axs, ax2)
        ax2.yaxisposition = :right
        ax2.yticklabelalign = (:left, :center)
        ax2.xticklabelsvisible = false
        ax2.xticklabelsvisible = false
        ax2.xlabelvisible = false
    end
    linkyaxes!(axs...)

    f
end

#@recipe(PlotOEDSolution, oedresult) do scene
#    Theme(
#        plot_color = :red
#    )
#end
#
#function Makie.plot!(p::PlotOEDSolution)
#    oedresult = p[:oedresult][]
#
#
#    for (i, g) in enumerate(gain)
#        lines!(p[1,i], x, g, color = p[:plot_color][], label="trace Π$i(t)", xlabel="Time")
#        hlines!(p, oedresult.multiplier[i], linestyle=:dash, label="μ$i")
#    end
#    p
#end