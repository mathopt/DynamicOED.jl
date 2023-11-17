using ModelingToolkit
using DynamicOED
using Test

## Define the systems
@variables t
@variables x(t)=2.0 [description = "State with uncertain initial condition", tunable=false, bounds = (0.1, 5.0)] # The first state is uncertain
@parameters p[1:1]=-2.0 [description = "Fixed parameter", tunable = true] 
@variables y₁(t) [description = "Observed", measurement_rate = 1.0] 
@variables y₂(t) [description = "Observed 2", measurement_rate = 0.3] 
@parameters c=0.0 [description = "Control", input = true, measurement_rate = 0.1] 
D = Differential(t)

# One observed
@named simple_system_1 = ODESystem(
    D.(x) .~  p .* x
, tspan = (0.0, 2.0),
observed = [
y₁ ~ first(x)]
)

@named oed_1 = OEDSystem(simple_system_1)
oed_1 = structural_simplify(oed_1)
prob_1 = DynamicOED.OEDProblem(oed_1, DCriterion())

#  Two observed
@named simple_system_2 = ODESystem(
    D.(x) .~  p .* x
, tspan = (0.0, 2.0),
observed = [
y₁ ~ first(x), 
y₂ ~ sqrt(x)]
)

@named oed_2 = OEDSystem(simple_system_2)
oed_2 = structural_simplify(oed_2)
prob_2 = DynamicOED.OEDProblem(oed_2, DCriterion())

# Two observed + control
@named simple_system_3 = ODESystem(
    D.(x) .~  p .* x .+ c
, tspan = (0.0, 2.0),
observed = [
y₁ ~ first(x), 
y₂ ~ sqrt(x)]
)

@named oed_3 = OEDSystem(simple_system_3)

oed_3 = structural_simplify(oed_3)
prob_3 = DynamicOED.OEDProblem(oed_3, DCriterion())

@testset "Univariate" begin 
    timegrid = prob_1.timegrid
    @test timegrid.variables == (Symbol("w₁"), )
    @test length(timegrid.timegrids) == 1
    @test timegrid.timegrids[1] == [(0., 1.), (1., 2.)]
    @test timegrid.timespans == [(0., 1.), (1., 2.)]
    @test DynamicOED.get_tspan(timegrid, 1) == (0., 1.)
    @test DynamicOED.get_variable_idx(timegrid, Symbol("w₁"),1)  == 1 
    @test DynamicOED.get_variable_idx(timegrid, Symbol("w₁"),2)  == 2 
end


@testset "Two observed" begin 
    timegrid = prob_2.timegrid
    @test timegrid.variables == (Symbol("w₁"), Symbol("w₂"))
    @test length(timegrid.timegrids) == 2
    @test timegrid.timegrids[1] == prob_1.timegrid.timegrids[1]
    @test timegrid.timegrids[2] == [(0., .3), (.3, .6), (.6, .9), (.9, 1.2), (1.2, 1.5), (1.5, 1.8), (1.8, 2.0)]
    @test timegrid.timespans != timegrid.timegrids[1] != timegrid.timegrids[2] 
    @test size(timegrid.timespans) == (8,) # We have
    @test timegrid.timespans == [(0., .3), (.3, .6), (.6, .9), (.9, 1.),(1., 1.2),  (1.2, 1.5), (1.5, 1.8), (1.8, 2.0)]
    # Right assignments
    @test DynamicOED.get_variable_idx(timegrid, Symbol("w₁"),4)  == 1
    @test DynamicOED.get_variable_idx(timegrid, Symbol("w₂"),4)  == 4
    @test DynamicOED.get_variable_idx(timegrid, Symbol("w₁"),5)  == 2
    @test DynamicOED.get_variable_idx(timegrid, Symbol("w₂"),5)  == 4
end


@testset "Two Observed and Control" begin 
    timegrid = prob_3.timegrid
    @test timegrid.variables  == (Symbol("c"), Symbol("w₁"), Symbol("w₂"),)
    @test length(timegrid.timegrids) == 3
    @test timegrid.timegrids[1] == [(t_i,t_j) for (t_i, t_j) in zip(0.0:0.1:1.9, 0.1:0.1:2.0)]
    @test timegrid.timegrids[2] == prob_2.timegrid.timegrids[1]
    @test timegrid.timegrids[3] == prob_2.timegrid.timegrids[2]
    @test timegrid.timespans != timegrid.timegrids[2] != timegrid.timegrids[3] 
    @test size(timegrid.timespans) == (20,) # We have
    @test timegrid.timespans == [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0), (1.0, 1.1), (1.1, 1.2), (1.2, 1.3), (1.3, 1.4), (1.4, 1.5), (1.5, 1.6), (1.6, 1.7), (1.7, 1.8), (1.8, 1.9), (1.9, 2.0)]
    # Right assignments
    @test DynamicOED.get_variable_idx(timegrid, Symbol("w₁"),4)  == 1
    @test DynamicOED.get_variable_idx(timegrid, Symbol("w₁"),10) == 1
    @test DynamicOED.get_variable_idx(timegrid, Symbol("w₁"),11) == 2
    @test DynamicOED.get_variable_idx(timegrid, Symbol("w₂"),1)  == 1
    @test DynamicOED.get_variable_idx(timegrid, Symbol("w₂"),10) == 4
    @test DynamicOED.get_variable_idx(timegrid, Symbol("w₂"),15) == 5
    @test DynamicOED.get_variable_idx(timegrid, Symbol("c"),1)  == 1
    @test DynamicOED.get_variable_idx(timegrid, Symbol("c"),10) == 10
    @test DynamicOED.get_variable_idx(timegrid, Symbol("c"),15) == 15
end