function SciMLBase.solve(ed::ExperimentalDesign, M::Int, criterion::AbstractInformationCriterion, solver, options; integer = false, ad_backend = AD.ForwardDiffBackend(), kwargs...)
    # Define the loss and constraints


    n_exp = length(ed.tgrid)
    n_vars = sum(ed.w_indicator)

    loss(w) = criterion(ed, reshape(w, n_vars, n_exp); kwargs...)
    
    m_constraints(w) = begin
        sol = last(ed(reshape(w, n_vars, n_exp); kwargs...))
        sum(sol[:,end-n_vars+1:end]) .- M
    end
    
    # Define a nonconvex model
    w_init = begin
        w = rand(Float64, n_vars, n_exp)
        foreach(eachcol(w)) do wi
            wi ./= sum(wi)
        end
        vec(w)
    end

    w_init = zeros(Float64, n_vars*n_exp)
    idxs = rand(1:n_vars*n_exp, M)
    w_init[idxs] .= one(Float64)
    model = Nonconvex.Model(loss)
    addvar!(model, zeros(Float64, n_vars*n_exp), ones(Float64, n_vars*n_exp), integer = integer ? ones(Bool, n_vars*n_exp) : zeros(Bool, n_vars*n_exp))
    add_ineq_constraint!(model, m_constraints)
    ## Convert to the backend
    ad_model = abstractdiffy(model, ad_backend)
    # Solve
    Nonconvex.optimize(ad_model, solver, w_init, options = options)
end