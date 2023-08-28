function build_extended_dynamics(prob::ODEProblem)
    f = prob.f

    observed_f = (f.observed != SciMLBase.DEFAULT_OBSERVED) ? f.observed : (u,p,t) -> u

    nx = length(prob.u0)
    np = length(prob.p)

    x = make_variables(:u, nx)
    p = make_variables(:p, np)
    t = make_variables(:t, 1)

    eq = f(x, p, t)
    G = make_variables(:G, nx,np)
    dfdx = FastDifferentiation.jacobian(eq, x)

    dfdp = FastDifferentiation.jacobian(eq, p)

    Ġ = dfdx * G .+ dfdp
    Σ = vcat(eq, vec(Ġ))

    states = vcat(x, vec(G))
    state_syms = (isa(f, ODEFunction) && !isnothing(f.syms)) ? vcat(f.syms... , Symbol.(vec(G))) : Symbol.(states)
    parameters = p

    h = observed_f(states[1:nx], parameters, t)
    W = make_variables(:w, length(h))
    hx = FastDifferentiation.jacobian(h, states[1:nx])
    fidxs = tril!(trues((np,np)))
    fvec = sum(map(enumerate(W)) do (i, wi)
        hxiG = hx[i:i,:] * G
        wi * ((hxiG'hxiG)[fidxs])
    end)

    jac_full = make_function(FastDifferentiation.jacobian(Σ, states), states, parameters, t; in_place = false)
    eq_full = make_function(Σ, states, parameters, t; in_place = false)
    f_full = make_function(fvec, states, parameters, t, W; in_place = false)

    # In place does not work as expected here, so we simply wrap
    f_new = let eq_full = eq_full
        (u, p, t) -> begin
            eq_full(vcat(u, p, t))
        end
    end
    jac_new = let jac_full = jac_full
        (u, p, t) -> begin
            jac_full(vcat(u, p, t))
        end
    end

    f_new_observed = let f_full = f_full
        (u, p, w, t) -> begin
            f_full(vcat(u, p, t, w))
        end
    end

    ODEFunction{false}(f_new, jac = jac_new,
        syms = state_syms, observed = f_new_observed
    )
end
