# Supervised methods
In order to use the supervised evaluation method you have to have a ground truth. 
```julia
function evaluate(cfg, cl::Matrix{T}, gt::Matrix{T})  where T<:Integer
```

## Example
```@example supervised
using TestImages, ImageSegmentation, Plots, Colors,  Images, ImageSegmentationEvaluation
using Plots
function get_random_color(seed)
    srand(seed)
    rand(RGB{N0f8})
end


img = Gray.(testimage("house"));
segments1 = felzenszwalb(img, 300, 100);
segments2 = felzenszwalb(img, 150, 200);

plt1 = Plots.plot(map(i->get_random_color(i), labels_map(segments1)))
plt2 = Plots.plot(map(i->get_random_color(i), labels_map(segments2)))
Plots.plot(plt1,plt2, layout=grid(1,2))
savefig("images.png"); nothing # hide
```

![](images.png)


- Segmentation Covering
- Variation of Information
- Rand index
- F-measure for regions
- Precision-recall for objects and parts
- Boundary displacement error
- F-measure for boundaries

```@example supervised
labels_1 = labels_map(segments1)
labels_2 = labels_map(segments2)

criteria = [Precision(), 
            FMeasure(), 
            SegmentationCovering(), 
            VariationOfInformation(true), 
            RandIndex(), 
            FMeasureRegions(), 
            PRObjectsAndParts(0.9, 0.25 , 0.1),
            BoundaryDisplacementError(),
            FBoundary(0.0075)]
names = ["Precision", "F-measdure", "SegCov", "VoI", "Rand", "F_r", "F_op", "BDE", "F_b"]
values = [ImageSegmentationEvaluation.evaluate(c, labels_2, labels_1) for c in criteria]
vcat(reshape(["Measure", "value"],1,2),hcat(names,first.(values)))
```
