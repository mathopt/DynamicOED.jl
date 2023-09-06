function build_extended_dynamics(prob::DAEProblem; parameters=1:length(prob.p), variable_iv=false)
    f = prob.f

    observed_f = (f.observed != SciMLBase.DEFAULT_OBSERVED) ? f.observed : (u,p,t) -> u

    FastDifferentiation.clear_cache()

    nx = length(prob.u0)
    np_original = length(prob.p)
    np_selected = length(parameters)
    np_iv = nx

    np_real = variable_iv ? np_selected + np_iv : np_selected
    np_created = variable_iv ? np_original + np_iv : np_original # just for dimension purposes

    x   = make_variables(:u, nx)
    dx  = make_variables(:du, nx)
    p = make_variables(:p, np_created)
    t   = make_variables(:t, 1)

    eq = f(dx, x, p, t)
    dG = make_variables(:dG, nx, np_real)
    G  = make_variables(:G, nx,np_real)
    G_iv =  zeros(nx, np_selected)
    G_iv = variable_iv ? hcat(G_iv, Matrix{eltype(G_iv)}(I,nx,nx)) : G_iv

    dfddx = FastDifferentiation.jacobian(eq, dx)
    dfdx = FastDifferentiation.jacobian(eq, x)
    parameters_considered = variable_iv ? vcat(parameters, collect(np_original+1:np_created)) : parameters
    dfdp = FastDifferentiation.jacobian(eq, p[parameters_considered])

    Ġ = dfddx * dG .+ dfdx * G .+ dfdp

    Σ = vcat(eq, vec(Ġ))
    dstates = vcat(dx, vec(dG))
    states = vcat(x, vec(G))
    state_syms = (isa(f, DAEFunction) && !isnothing(f.syms)) ? vcat(f.syms... , Symbol.(vec(G))) : Symbol.(states)

    h_ = observed_f(states[1:nx], p, t)
    h  = make_function(h_, states[1:nx], p, t)
    nh = length(h_)
    W = make_variables(:w, nh)
    hx = FastDifferentiation.jacobian(h_, states[1:nx])
    hx_fun = make_function(hx, states[1:nx], p, t)
    fidxs = tril!(trues((np_real,np_real)))

    fvec = sum(map(enumerate(W)) do (i, wi)
        hxiG = hx[i:i,:] * G
        wi * ((hxiG' * hxiG)[fidxs])
    end)


    eq_full = make_function(Σ, dstates, states, p, t; in_place = false)
    dF = make_function(fvec, states, p, t, W; in_place = false)

    # In place does not work as expected here, so we simply wrap
    f_new = let eq_full = eq_full
        (du, u, p, t) -> begin
            eq_full(vcat(du, u, p, t))
        end
    end

    integrand = let dF = dF
        (u, p, t, w_) -> begin
            dF(vcat(u, p, t, w_))
        end
    end

    new_f = DAEFunction{false}(f_new,# jac = jac_new,
        syms = state_syms
    )

    p = variable_iv ? vcat(prob.p, prob.u0) : prob.p

    new_prob = DAEProblem(new_f, vcat(prob.du0, vec(zeros(size(G_iv)))), vcat(prob.u0, vec(G_iv)), prob.tspan, p)

    FastDifferentiation.clear_cache()

    nxnh = (nx=nx, nh=nh, np=np_real)
    observed = (h=h, hx=hx_fun,)
    OEDFisher(new_prob, integrand, observed, nxnh)
end
