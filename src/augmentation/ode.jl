function build_extended_dynamics(prob::ODEProblem)
    f = prob.f

    observed_f = (f.observed != SciMLBase.DEFAULT_OBSERVED) ? f.observed : (u,p,t) -> u    
    x = make_variables(:u, length(prob.u0))
    p = make_variables(:p, length(prob.p))
    t = first(make_variables(:t))



    eq = f(x, p, t)
    G = make_variables(:G, size(eq,1), size(p,1))
    dfdx = FastDifferentiation.jacobian(eq, x)
    
    dfdp = FastDifferentiation.jacobian(eq, p)
    
    Ġ = dfdx * G .+ dfdp
    Σ = vcat(eq, vec(Ġ))
    
    states = vcat(x, vec(G))
    state_syms = (isa(f, ODEFunction) && !isnothing(f.syms)) ? vcat(f.syms... , Symbol.(vec(G))) : Symbol.(states) 
    parameters = p
    time = [t]
    
    h = observed_f(states, parameters, time)
    W = make_variables(:w, length(h))
    hx = FastDifferentiation.jacobian(h, states)
    @info hx
    fidxs = tril!(trues((length(p), length(p))), -1)
    fvec = sum(map(zip(W, eachrow(hx))) do (wi, hxi)
        x = vec(vec(hxi)*G)
        wi.*((x'x)[fidxs])
    end)#(hx*G)'*diag(W)*(hx*G)
    #fvec = dFdt[tril!(trues(size(dFdt)), -1)]
    @info h
    @info dFdt
    @info fvec

    jac_full = make_function(FastDifferentiation.jacobian(Σ, states), states, parameters, time; in_place = false)
    eq_full = make_function(Σ, states, parameters, time; in_place = false)
    f_full = make_function(fvec, states, parameters, time, W; in_place = false)

    # In place does not work as expected here, so we simply wrap 
    f_new = let eq_full = eq_full
        (du, u, p, t) -> begin 
            du .= eq_full(vcat(u, p, t))
            nothing
        end
    end
    jac_new = let jac_full = jac_full
        (J, u, p, t) -> begin 
            J .= jac_full(vcat(u, p, t))
            nothing
        end
    end

    f_new = let f_full = f_full
        (u, p, t, w) -> begin 
            f_full(vcat(u, p, t, w))
        end
    end

    ODEFunction{true}(f_new, jac = jac_new, jac_prototype = zeros(eltype(prob.u0), size(eq,1), size(states, 1)), 
        syms = state_syms, observed = f_new
    )
end

function build_extended_problem(prob::ODEProblem)
    dudt = build_extended_dynamics(prob)

    u0 = prob.u0
    p = prob.p

    u0_ = vcat(copy(u0), vec(zeros(eltype(u0), size(u0, 1), size(p, 1))))
    ODEProblem(dudt, u0_, prob.tspan, p;  prob.kwargs...)
end

function build_hx(prob::ODEProblem)

    u0 = prob.u0
    p = prob.p

end