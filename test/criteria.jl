using LinearAlgebra
using DynamicOED
using Test

@testset "Conversion to symmetric matrix from vector" begin
    for i=1:20
        A = Symmetric(rand(0:.1:100,i,i))
        idxs = triu(ones(size(A))) .== 1.0
        @test DynamicOED._symmetric_from_vector(A[idxs]) == A
        @test DynamicOED._symmetric_from_vector(A[idxs], Val(i)) == A
    end
end

@testset "Criteria" begin

    F = [5.68145   -0.148312   2.00382   5.1841      3.57814
        -0.148312   2.17276   -1.18576   1.57651    -3.95544
         2.00382   -1.18576    3.8703   -0.14832     2.69545
         5.1841     1.57651   -0.14832  10.9617      0.0774284
         3.57814   -3.95544    2.69545   0.0774284  24.8495]

    test_results = [(FisherACriterion(), -47.53571), (FisherDCriterion(), -2180.4107236837117),
                    (FisherECriterion(), -1.0326129713287733), (ACriterion(), 2.103928493261123),
                    (DCriterion(), 0.0004586291881332073), (ECriterion(), 0.9684170427504832)]

    for (crit, res) in test_results
        @test isapprox(crit(F,0.0), res)
    end
end