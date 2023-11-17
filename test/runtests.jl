using Test
using SafeTestsets

# Automatic Quality test
@safetestset "Aqua.jl" begin
    using Aqua
    using DynamicOED
    @testset "Project.toml" begin 
        Aqua.test_deps_compat(DynamicOED)
        Aqua.test_stale_deps(DynamicOED)
        Aqua.test_project_toml_formatting(DynamicOED)
        Aqua.test_project_extras(DynamicOED)
    end
    @testset "Piracy" Aqua.test_piracy(DynamicOED)
    @testset "Ambiguities" Aqua.test_ambiguities(DynamicOED)
    @testset "Unbounds" Aqua.test_unbound_args(DynamicOED)
    @testset "Undefined" Aqua.test_undefined_exports(DynamicOED)
end

@safetestset "MTK Extensions" begin 
    # Basic MTK extension tests
    include("./mtk_extensions.jl")
end

@safetestset "Discretize" begin 
    # Test size and assignment of timegrids
    include("./discretize.jl")
end

@safetestset "Criteria" begin 
    # References for all criteria
    include("./criteria.jl")
end
@testset "Optimization and Examples" begin 
    # Only simple test for all criteria
    @safetestset "1D" begin include("./references/1D.jl") end
    # Test for controls (integer + relaxed)
    #@safetestset "LotkaVolterra" begin end
    ## Test for DAE support
    #@safetestset "Rober" begin end
end