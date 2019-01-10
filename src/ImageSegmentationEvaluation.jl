module ImageSegmentationEvaluation
export
    evaluate,
    
    BoundaryGradient,
    BoundaryShift,
    BoundaryMooreTracing 
# package code goes here
include("utils.jl")
include("supervised.jl")


export unsupervised_metrics
include("unsupervised.jl")

end # module
