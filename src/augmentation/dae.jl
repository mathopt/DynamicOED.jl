function build_extended_dynamics(prob::DAEProblem; variable_iv=false)
    f = prob.f

    observed_f = (f.observed != SciMLBase.DEFAULT_OBSERVED) ? f.observed : (u,p,t) -> u

    FastDifferentiation.clear_cache()

    nx = length(prob.u0)
    np_or = length(prob.p)
    np_iv = nx

    np = variable_iv ? np_or + np_iv : np_or

    x   = make_variables(:u, nx)
    dx  = make_variables(:du, nx)
    p   = make_variables(:p, np)
    t   = make_variables(:t, 1)

    eq = f(dx, x, p, t)
    dG = make_variables(:dG, nx, np)
    G = make_variables(:G, nx,np)
    G_iv =  zeros(nx,np_or)
    G_iv = variable_iv ? hcat(G_iv, Matrix{eltype(G_iv)}(I,nx,nx)) : G_iv

    dfddx = FastDifferentiation.jacobian(eq, dx)
    dfdx = FastDifferentiation.jacobian(eq, x)
    dfdp = FastDifferentiation.jacobian(eq, p)

    Ġ = dfddx * dG .+ dfdx * G .+ dfdp

    Σ = vcat(eq, vec(Ġ))
    dstates = vcat(dx, vec(dG))
    states = vcat(x, vec(G))
    state_syms = (isa(f, DAEFunction) && !isnothing(f.syms)) ? vcat(f.syms... , Symbol.(vec(G))) : Symbol.(states)
    parameters = p

    h = observed_f(states[1:nx], parameters, t)
    nh = length(h)
    W = make_variables(:w, nh)
    hx = FastDifferentiation.jacobian(h, states[1:nx])
    fidxs = tril!(trues((np,np)))

    fvec = sum(map(enumerate(W)) do (i, wi)
        hxiG = hx[i:i,:] * G
        wi * ((hxiG' * hxiG)[fidxs])
    end)


    eq_full = make_function(Σ, dstates, states, parameters, t; in_place = false)
    dF = make_function(fvec, states, parameters, t, W; in_place = false)

    # In place does not work as expected here, so we simply wrap
    f_new = let eq_full = eq_full
        (du, u, p, t) -> begin
            eq_full(vcat(du, u, p, t))
        end
    end

    f_new_observed = let dF = dF
        (u, p, t, w_) -> begin
            dF(vcat(u, p, t, w_))
        end
    end

    new_f = DAEFunction{false}(f_new,# jac = jac_new,
        syms = state_syms
    )

    p = variable_iv ? vcat(prob.p, prob.u0) : prob.p

    new_prob = DAEProblem(new_f, vcat(prob.du0, vec(zeros(nx,np))), vcat(prob.u0, vec(G_iv)), prob.tspan, p)

    FastDifferentiation.clear_cache()

    nxnh = (nx=nx, nh=nh)

    OEDFisher(new_prob, f_new_observed, nxnh)
end
