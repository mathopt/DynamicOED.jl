using Test
using SafeTestsets

@safetestset "Criteria API" begin include("./criteria.jl") end
@safetestset "Experimental Design" begin include("./experimental_design.jl") end
@safetestset "Reference results" begin include("./references.jl") end