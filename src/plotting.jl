function remove_excess_parentheses_and_whitespace(str::String)
    idx1 = first(findfirst("(", str))
    idx1 == first(findlast("(", str)) && return str # only remove outermost set of parentheses
    idx2 = first(findlast(")", str))

    str = str[[i for i=1:length(str) if i != idx1 && i != idx2]]
    filter(x -> !isspace(x), str)
end


function plot_ode_sol!(f::Figure, res::OEDSolution; kwargs...)

    first_plot = isempty(contents(f.layout))
    current_layout = f.layout.size
    idx = first_plot ? 1 : current_layout[1] + 1

    ax = Axis(f[idx,1], title="State Trajectory", xlabel="t")
    idxs = states(structural_simplify(res.oed.sys_original))
    x = reduce(hcat, map(enumerate(res.sol)) do (i,sol)
        i < length(res.sol) ? reduce(hcat, sol[idxs][1:end-1]) : reduce(hcat, sol[idxs])
    end)

    foreach(enumerate(eachrow(x))) do (i,xi)
        lines!(ax, res.t, xi, cycle = [:color], label = string.(idxs[i]); kwargs...)
    end
    leg = Legend(f[idx,2], ax)
    nothing
end

function plot_sensitivities!(f::Figure, res::OEDSolution; kwargs...)

    first_plot = isempty(contents(f.layout))
    current_layout = f.layout.size
    idx = first_plot ? 1 : current_layout[1] + 1

    ax = Axis(f[idx,1], title="Sensitivity", xlabel="t")

    labels = string.(vec(res.oed.variables.G)) .|> remove_excess_parentheses_and_whitespace
    foreach(enumerate(eachrow(res.sensitivities))) do (i, g)
        lines!(ax, res.t, g, label = labels[i]; kwargs...)
    end
    leg = Legend(f[idx,2], ax, nbanks = maximum([length(res.oed.variables.G) ÷ 6, 1]))
    nothing
end

function plot_observed!(f::Figure, res::OEDSolution; kwargs...)

    first_plot = isempty(contents(f.layout))
    current_layout = f.layout.size
    idx = first_plot ? 1 : current_layout[1] + 1

    ax = Axis(f[idx,1], title="Observed", xlabel="t")
    h = res.oed.variables.h
    obs = reduce(hcat, map(enumerate(res.sol)) do (i,sol)
        i < length(res.sol) ? reduce(hcat, sol[h][1:end-1]) : reduce(hcat, sol[h])
    end)

    labels_observed = collect(h) .|> string .|> remove_excess_parentheses_and_whitespace
    foreach(enumerate(eachrow(obs))) do (i, row)
        lines!(ax, res.t, row, label=labels_observed[i]; kwargs...)
    end
    leg = Legend(f[idx,2], ax, nbanks = maximum([length(res.oed.variables.G) ÷ 6, 1]))

    nothing
end

function plot_sampling!(f::Figure, res::OEDSolution; kwargs...)

    first_plot = isempty(contents(f.layout))
    current_layout = f.layout.size
    idx = first_plot ? 1 : current_layout[1] + 1

    ax = Axis(f[idx,1], title="Sampling", xlabel="t")
    tspan       = (first(res.sol).t[1], last(res.sol).t[end])
    Δt          = -(reverse(tspan)...)/length(res.sol)
    tControl    = first(tspan):Δt:last(tspan)

    foreach(enumerate(eachrow(res.w.w))) do (i,w)
        stairs!(ax, tControl, [w[1]; w], label="w$i(t)", stairs=:pre; kwargs...)
    end
    leg = Legend(f[idx,2], ax, nbanks = maximum([length(res.oed.variables.G) ÷ 6, 1]))

    nothing
end

function plot_sampling_and_opt_crit!(f::Figure, res::OEDSolution; kwargs...)

    first_plot = isempty(contents(f.layout))
    current_layout = f.layout.size
    idx = first_plot ? 1 : current_layout[1] + 1

    # Generate a plot for each measurement
    idxs        = sort(string.(parameters(res.oed.sys))[end-size(res.w.w, 1)+1:end])
    tspan       = (first(res.sol).t[1], last(res.sol).t[end])
    Δt          = -(reverse(tspan)...)/length(res.sol)
    tControl    = first(tspan):Δt:last(tspan)

    opt_crit, label_criterion     = switching_function(res)
    n_vars = sum(res.oed.w_indicator)

    # Check the grid layout in advance
    n_rows = max(ceil(Int, size(res.w.w,1) / 2), 1)
    n_cols = max(ceil(Int, size(res.w.w,1) / n_rows), 1)
    current_idx = 1

    for i in 1:n_rows
        if i==1
            title_axis = Axis(f[idx,1], title="Sampling")
            hidedecorations!(title_axis)
            hidespines!(title_axis)
        end
        yaxs = []
        for j in 1:n_cols
            current_idx > size(res.w.w,1) && break

            ax_1 = Axis(f[idx,1][i,j], xlabel = "t", subtitle = idxs[current_idx])
            ax_2 = Axis(f[idx,1][i,j], yticklabelcolor=:red, yaxisposition=:right, yminorgridvisible = false, ygridvisible = false)

            p1 = stairs!(ax_1, tControl, [res.w.w[current_idx,1];res.w.w[current_idx,:]], label = idxs[current_idx], color = :black, stairs = :pre)
            p2 = lines!(ax_2, res.t, opt_crit[current_idx], color = :red, grid = false)
            p3 = hlines!(ax_2, res.multiplier[current_idx], color=:red, linestyle=:dash)

            linkxaxes!(ax_1, ax_2)
            push!(yaxs, ax_2)
            ax_2.yaxisposition = :right
            ax_2.yticklabelalign = (:left, :center)
            ax_2.xticklabelsvisible = false
            ax_2.xlabelvisible = false
            current_idx += 1
            j == 1 && current_idx <= size(res.w.w,1) && hideydecorations!(ax_2)
            ylims!(ax_1, -.05, 1.05)
            j == 2 && hideydecorations!(ax_1)


            if current_idx == n_vars
                ps = [p1, p2, p3]
                ls = ["w(t)", label_criterion, "μ"]
                leg = Legend(f[idx,2], ps, ls)
            end

        end
        linkyaxes!(yaxs...)
    end

    !first_plot && rowsize!(f.layout, idx, Relative(n_rows/(n_rows+current_layout[1])))
    nothing
end

"""
$(SIGNATURES)

Plots the results from the `OEDProblem`, e.g., differential states, sampling function, etc. in
a `Makie.Figure`, which can be provided via `f`.

If `observed` is set to `true`, the observed variables defined in the underlying `ODESystem` are plotted.

If `sensitivities` is set to `true`, the sensitivities are plotted.

If `opt_crit` is set to `true`, the sufficient conditions for whether the sampling is on are plotted.
These are depending on the criterion and the Lagrange multiplier of the measurement constraint.
"""
function plotOED(res::OEDSolution; f = Figure(resolution=(1200,1200)), observed::Bool=true,
                sensitivities::Bool=true, opt_crit::Bool=false, kwargs...)

    plot_ode_sol!(f, res; kwargs...)
    observed && plot_observed!(f, res; kwargs...)
    sensitivities && plot_sensitivities!(f, res; kwargs...)
    (!opt_crit || (opt_crit && isnothing(res.multiplier))) && plot_sampling!(f, res; kwargs...)
    opt_crit && !isnothing(res.multiplier) && plot_sampling_and_opt_crit!(f, res)
    trim!(f.layout)
    f
end
