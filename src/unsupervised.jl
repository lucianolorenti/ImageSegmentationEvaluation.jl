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

#Unsupervised Performance Evaluation of Image Segmentation

"""
Zéboudj, Rachid. Filtrage, seuillage automatique, contraste et contours: du pré-traitement à l'analyse d'image. Diss. Saint-Etienne, 1988.
Unsupervised Evaluation of Image Segmentation Application to Multi-spectral Images

"This contrast takes into account the internal and external contrast of the regions measured in the neighborhood of each pixel"
"""
type Zeboudj
end

function evaluate(c::Zeboudj, image::Matrix{T1}, segments::Matrix{T}) where T<:Integer where T1<:Number
    local N = maximum(segments)
    local inside = zeros(N)
    local outside = zeros(N)
    local segments_sizes = zeros(N)
    local border_lengths = zeros(N)
    I1, Iend = first(R), last(R)
    local W = CartesianIndex(r,r)
    for I in CartesianRange(size(image))
        local current_label = segments[I]
        segment_sizes[curret_label]+=1
        local max_inside = -1
        local max_outside = -1
        local is_a_border_point = false
        for J in CartesianRange(max(I1, I-W), min(Iend, I+W))
            if current_label != segments[J]
                is_a_border_point = true
                max_outside = max(maxOutside, image[J])
            else
                max_inside = max(maxInside, image[J])
            end            
        end
        if (is_a_border_point)
            border_lengths[current_label]+=1
        end
        inside[current_label]+=max_inside
        outside[current_label]+=max_outside;
    end
    inside=inside/segments_sizes
    outside=outside/border_lengths

    local C = zeros(N)
    for j=1:N
        if (inside[j] <0 ) &&  (inside[j] < outside[j])
            C[j]= 1- (inside[j]/outside[j])
        elseif (inside[j]==0)
            C[j]= outside[j]
        else
            C[j]= 0
        end
    end
    return (1/(prod(size(image)))) *sum(segments_sizes.*C)
end
"""
An Entropy-based Objective Evaluation Method for Image Segmentation
Hui Zhang*, Jason E. Fritts and Sally A. Goldman
"""
type ValuesEntropy
end
function evaluate(c::ValuesEntropy, image::Matrix, segments::Matrix{T}) where T<:Integer
    
end
