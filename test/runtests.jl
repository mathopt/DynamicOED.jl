using Test
using SafeTestsets

@safetestset "Reference results" begin include("./references.jl") end