using LinearAlgebra
using DynamicOED

@testset "Conversion to symmetric matrix from vector" begin
    for i=1:20
        A = Symmetric(rand(0:.1:100,i,i))
        idxs = triu(ones(size(A))) .== 1.0
        @test DynamicOED._symmetric_from_vector(A[idxs]) == A
        @test DynamicOED._symmetric_from_vector(A[idxs], i) == A
    end
end
