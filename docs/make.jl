using Documenter

push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using DynamicOED

makedocs(
    modules = [DynamicOED], 
    sitename = "DynamicOED.jl", 
    remotes = nothing,
    draft = false,
    pages = [
        "Home" => "index.md", 
        "Examples" => [
            "Linear System" => "examples/1D.md",
        ],
        "Theory" => "theory.md",
        "API" => "api.md"
    ]
)