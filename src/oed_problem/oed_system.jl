using Revise
using DynamicOED
using Integrals
using OrdinaryDiffEq

u0 = [0.5,0.7]
tspan = (0.0,1.0)

function lotka(u, p, t)
    x,y = u
    return [x - x*y, -y + x*y]
end

prob = ODEProblem(lotka, u0, tspan, [])

DynamicOED.augment_problem(prob)