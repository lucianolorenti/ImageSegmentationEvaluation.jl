# On Selecting the Best Unsupervised Evaluation Techniques for Image Segmentation
"""
The use of visible color difference in the quantitative evaluation of color image segmentation,
"""
struct ECW
    threshold::Float64
end
function evaluate(c::ECW, image::Matrix{LAB}, segments::Matrix{T}) where T<:Integer
    local segments_mean = [i->segment_mean(segments,i) for i in unique(segments)]
    local R = length(segments_mean)
    local mean_segments = map(i->segments_mean[i],segments)
    local E_intra = sum(norm.(image-mean_segments).<  c.threshold)
    local C = 1/6
    local sum = 0
    for i=1:R-1
        local a = segments.==i
        for j=i+1:R
            local b = segments.==j            
            if (norm(segments_mean[i] - segments_mean[j])  > c.threshold)                
                local Kij = sum((a[1:end-1,1:end] + b[2:end,1:end]).==2)  + sum((a[1:end,1:end-1] + b[1:end,2:end-1]).==2)
                sum+=Kij                
            end
        end
    end
    local E_inter= (2*sum)/(C*prod(size(segmnets)))
    return 0.5*(E_intra +  E_inter) 
    
end
function evaluate(c::ECW, image::Matrix{RGB}, segments::Matrix{T}) where T<:Integer
    return evaluate(c, LAB.(image), segments)
end
