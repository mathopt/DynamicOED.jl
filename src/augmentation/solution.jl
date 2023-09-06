"""
$(TYPEDEF)

The solution to an optimal experimental design problem for a certain criterion.
The sampling decisions can be retrieved from `w`.

# Fields
$(FIELDS)
"""
struct OEDSolution{C,S,F,W,I,M,O} <: AbstractOEDSolution
    "Criterion"
    criterion::C
    "The solution of the system of ODEs"
    sol::S
    "Fisher information at end of time horizon"
    F_tf::F
    "Optimal sampling solution"
    w::W
    "Information gain matrices"
    information_gain::I
    "Lagrange multipliers corresponding to sampling constraint"
    multiplier::M
    "Objective"
    obj::O
    "Experimental design"
    oed::AbstractExperimentalDesign
end

function OEDSolution(oed::AbstractExperimentalDesign, criterion::AbstractInformationCriterion,
                     w::W, obj::Real; μ=nothing, kwargs...) where W

    sol             = get_t_and_sols(oed, w; kwargs...)
    F_tf            = oed(w; kwargs...)
    P               = compute_local_information_gain(oed, sol.u, sol.t);
    Π               = compute_global_information_gain(oed, F_tf, P);

    information_gain = (local_information_gain = P, global_information_gain=Π,)

    return OEDSolution{typeof(criterion), typeof(sol), typeof(F_tf), typeof(w), typeof(information_gain), typeof(μ), typeof(obj)}(
        criterion, sol, F_tf, w, information_gain, μ, obj, oed
    );
end