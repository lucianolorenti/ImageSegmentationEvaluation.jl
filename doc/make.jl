using Documenter, ImageSegmentationEvaluation

makedocs(
  format = :html,
    sitename = "Image Segmentation Evaluation",
    pages = [
        "Index"=> "index.md",
        "Subsection" => [
            "Supervised methods" => "supervised.md",
            "Unsupervised methods" => "unsupervised.md"
        ]
    ]
)

