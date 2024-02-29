using ModelingToolkit
using LinearAlgebra
using DynamicOED

# Define the system
T_min, T_max = 293.16, 473.16

const M = [0.11911, 0.07412, 0.19323, 0.31234, 0.35733, 0.07806]
const rho = [1095.0, 809.0, 1415.0, 1528.0, 1451.0, 1101.0]

K_REF1 = 5.0E-4
E_A1 = 35240.0
K_REF2 = 8.0E-8
E_A2 = 85000.0
K_REF4 = 1.0E-8
E_A4 = 35000.0
DH_2 = -17031.0
K_C2 = 0.17

R = 8.314;
T1 = 363.16;

@variables t
@variables V(t)=0.5 [description = "Reaction volume"]
@variables r1(t)=0.0 [description = "Reaction rate 1"]
@variables r2(t)=0.0 [description = "Reaction rate 2"]
@variables r3(t)=0.0 [description = "Reaction rate 3"]
@variables r4(t)=0.0 [description = "Reaction rate 4"]
@variables feed1(t)=0 [description = "State for feed 1"]
@variables feed2(t)=0 [description = "State for feed 2"]
@variables temperature(t)=293.15 [
    bounds = [293.16, 473.16],
    description = "State for temperature",
]
@variables n(t)[1:6]=[0.0; 0.0; 0.0; 0.0; 0.0; 0.0] [description = "States"]
@parameters u1 [
    description = "Control feed 1",
    bounds = [0, Inf],
    input = true,
    measurement_rate = 10,
]
@parameters u2 [
    description = "Control feed 2",
    bounds = [0, Inf],
    input = true,
    measurement_rate = 10,
]
@parameters u3 [
    description = "Control temperature",
    bounds = [-40, 40],
    input = true,
    measurement_rate = 10,
]
@variables h₁(t) [description = "Observed", measurement_rate = 20]
@variables h₂(t) [description = "Observed", measurement_rate = 20]
h = vcat(h₁, h₂)
@parameters p[1:6]=[1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0] [
    description = "Scaling parameters",
    tunable = true,
]
@parameters n0[1:3]=[0.12; 0.0; 0.0] [description = "Initial mole numbers", tunable = false]
D = Differential(t)

## Define the control function, returns feed1, feed2, T
eqs_ = [V * (r1 - r2 + r3);
    V * (r2 - r3);
    V * r4]

subs_ = Dict(r1 => p[1] * K_REF1 * exp(-p[2] * E_A1 / R * (1 / temperature - 1 / T1)) *
                   n[1] * n[2] / (V * V),
    r2 => p[3] * K_REF2 * exp(-p[4] * E_A2 / R * (1 / temperature - 1 / T1)) * n[1] * n[3] /
          (V * V),
    r3 => p[3] * K_REF2 * exp(-p[4] * E_A2 / R * (1 / temperature - 1 / T1)) *
          inv(K_C2 * exp(-(-DH_2 / R) * (1 / temperature - 1 / T1))) * n[4] / V,
    r4 => p[5] * K_REF4 * exp(-p[6] * E_A4 / R * (1 / temperature - 1 / T1)) * (n[1] / V)^2)

eqs = map(Base.Fix2(substitute, subs_), eqs_)

# Define the eqs
@named urethan = ODESystem([D(feed1) ~ u1;
        D(feed2) ~ u2;
        D(temperature) ~ u3;
        D(n[3]) ~ eqs[1];                                  #n_C
        D(n[4]) ~ eqs[2];                                  #n_D
        D(n[5]) ~ eqs[3];                                  #n_E
        n[1] ~ n0[1] + feed1 - n[3] - 2 * n[4] - 3 * n[5];    #n_A
        n[2] ~ n0[2] + feed2 - n[3] - n[4];                #n_B
        n[6] ~ n0[3] + feed1 + u2;                         #n_L
        V ~ sum(n .* M ./ rho)], tspan = (0.0, 80.0),
    observed = h .~ [100 * n[1] * M[1] / sum([ni .* M[i] for (i, ni) in enumerate(n)]);
    #100 * n[3]*M[3]/sum([ni .* M[i] for (i,ni) in enumerate(n)]);
    #100 * n[4]*M[4]/sum([ni .* M[i] for (i,ni) in enumerate(n)]);
        100 * n[5] * M[5] / sum([ni .* M[i] for (i, ni) in enumerate(n)])])

## Build the OED System

@named urethan_oed = OEDSystem(structural_simplify(urethan))

oed_problem = DynamicOED.OEDProblem(urethan_oed, DCriterion())

optimization_variables = states(oed_problem)
timegrids = DynamicOED.get_timegrids(oed_problem)
