using Documenter, ImageSegmentationEvaluation

makedocs(
    modules=[ImageSegmentationEvaluation],
    format = Documenter.HTML(prettyurls = true),
    sitename = "Image Segmentation Evaluation",
    source = "src",
    clean=false,
    pages = [
        "Index"=> "index.md",
        "Subsection" => [
            "Supervised methods" => "supervised.md",
            "Unsupervised methods" => "unsupervised.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/lucianolorenti/SpectralClustering.jl.git",
    deps = nothing,
    make = nothing,
    target = "build"
)

