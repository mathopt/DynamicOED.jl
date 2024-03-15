using ModelingToolkit
using DynamicOED

## Define the system
@variables t
@variables x(t)=2.0 [
    description = "State with uncertain initial condition",
    tunable = false,
    bounds = (0.1, 5.0)
] # The first state is uncertain
@parameters p[1:1]=-2.0 [description = "Fixed parameter", tunable = true]
@variables y₁(t) [description = "Observed", measurement_rate = 1.0]
D = Differential(t)

# Define the eqs

@named simple_system = ODESystem(D.(x) .~ p .* x, tspan = (0.0, 2.0),
    observed = [
        y₁ ~ first(x)])

@named oed = OEDSystem(simple_system)

@test isa(oed, ODESystem)

augmented_states = states(oed)
@test DynamicOED.is_fisher_state.(augmented_states) == Bool[false, false, true, false]
@test DynamicOED.is_information_gain.(augmented_states) == Bool[false, false, false, true]
@test size(augmented_states) == (4,)

augmented_parameters = parameters(oed)
@test DynamicOED.is_measurement_function.(augmented_parameters) == Bool[false, true]
@test size(augmented_parameters) == (2,)

augmented_equations = equations(oed)
augmented_observed = observed(oed)

@test size(augmented_equations) == (4,)
@test size(augmented_observed) == size(observed(simple_system))

G, F, Q = augmented_states[[2, 3, 4]]
w = augmented_parameters[2]

groundtruth_equations = ModelingToolkit.scalarize(vcat(D.(x) .~ p .* x,  # Sytem
    0.0 .~ x .- D.(G) .+ p .* G, # Sensitivity
    D.(F) .~ G .^ 2 .* w, # FIM DEQ
    Q .~ G))

@test all((x) -> isequal(x...), zip(augmented_equations, groundtruth_equations))
@test all((x) -> isequal(x...), zip(augmented_observed, observed(simple_system)))

reduced_oed = structural_simplify(oed)
simplified_equations = equations(reduced_oed)

@test size(simplified_equations) == (3,)
@test all(x -> isequal(x...), zip(states(reduced_oed), augmented_states[1:(end - 1)]))
@test size(observed(reduced_oed)) == (2,)
@test all((x) -> isequal(x...),
    zip(vcat(observed(oed), groundtruth_equations[end]), observed(reduced_oed)))

initials = ModelingToolkit.defaults(reduced_oed)
@test ModelingToolkit.defaults(oed) == initials
@test initials[x] == 2.0
@test initials[p[1]] == -2.0
@test initials[F] == 0.0
@test initials[G] == 0.0
@test initials[w] == 0.0

@test isa(ODAEProblem(reduced_oed, initials, (0.0, 10.0)), ODEProblem)
