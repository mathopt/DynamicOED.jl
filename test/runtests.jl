using Test
using SafeTestsets

@safetestset "Criteria API" begin include("./criteria.jl") end
@safetestset "Reference results" begin include("./references.jl") end