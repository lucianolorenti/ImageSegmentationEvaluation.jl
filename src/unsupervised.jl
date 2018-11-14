using Colors
# On Selecting the Best Unsupervised Evaluation Techniques for Image Segmentation
"""
The use of visible color difference in the quantitative evaluation of color image segmentation,
"""
struct ECW
    threshold::Float64
end
function evaluate(c::ECW, image::Matrix{Lab}, segments::Matrix{T}) where T<:Integer
     segments_mean = [i->segment_mean(segments,i) for i in unique(segments)]
     R = length(segments_mean)
     mean_segments = map(i->segments_mean[i],segments)
     E_intra = sum(norm.(image-mean_segments).<  c.threshold)
     C = 1/6
     sum = 0
    for i=1:R-1
         a = segments.==i
        for j=i+1:R
             b = segments.==j            
            if (norm(segments_mean[i] - segments_mean[j])  > c.threshold)                
                 Kij = sum((a[1:end-1,1:end] + b[2:end,1:end]).==2)  + sum((a[1:end,1:end-1] + b[1:end,2:end-1]).==2)
                sum+=Kij                
            end
        end
    end
     E_inter= (2*sum)/(C*prod(size(segmnets)))
    return 0.5*(E_intra +  E_inter)   
end

function evaluate(c::ECW, image::Matrix{RGB}, segments::Matrix{T}) where T<:Integer
    return evaluate(c, Lab.(image), segments)
end


"""
Zéboudj, Rachid. Filtrage, seuillage automatique, contraste et contours: du pré-traitement à l'analyse d'image. Diss. Saint-Etienne, 1988.
Unsupervised Evaluation of Image Segmentation Application to Multi-spectral Images

"This contrast takes into account the internal and external contrast of the regions measured in the neighborhood of each pixel"
"""
struct Zeboudj
end

function evaluate(c::Zeboudj, image::Matrix{T1}, segments::Matrix{T}) where T<:Integer where T1<:Number
     N = maximum(segments)
     inside = zeros(N)
     outside = zeros(N)
     segments_sizes = zeros(N)
     border_lengths = zeros(N)
    I1, Iend = first(R), last(R)
     W = CartesianIndex(r,r)
    for I in CartesianRange(size(image))
         current_label = segments[I]
        segment_sizes[curret_label]+=1
         max_inside = -1
         max_outside = -1
         is_a_border_point = false
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

    C = zeros(N)
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
@doc raw"""
An Entropy-based Objective Evaluation Method for Image Segmentation
Hui Zhang*, Jason E. Fritts and Sally A. Goldman

Given an image I of ($n \times m$) , $S_I = nm$

$H_v(R_j) = - \sum\limits_{m \in V^{(v)}_j} \dfrac{L_j(m)}{S_j) log(\dfrac{L_j(m)}{S_j)
)$

$ H_l(I) = - \sum\limits_{j=1}^N \dfrac{S_j}{S_I} log(\dfrac{S_j}{S_I}) $
"""
struct ValuesEntropy
end
function components(a::Vector{T}) where T<:Color
    return (comp1.(a), comp2.(a), comp3.(a))
end
function components(a::Vector{T}) where T<:TransparentColor    
    return (comp1.(a), comp2.(a), comp3.(a), alpha.(a))
end
function components(a::Vector{T}) where T<:Number
    return a
end
function evaluate(c::ValuesEntropy, image::Matrix, segments::Matrix{T}) where T<:Integer
    Hr = 0
    Hl = 0 
    Si = prod(size(image))
    for j in unique(segments)
        segment = segments.==j
        Sj = count(segment)
        comps = components(image[segment])
        Hl += (Sj/Si)*(log(Sj/Si))
        for k=1:length(comps)
            Ej = 0
            for val in unique(comps[k])
                Lj = count(comps[k].==val)
                Ej += (Lj / Sj)*(log(Lj / Sj))
            end
        end
        Hr += (Sj/Si)*(-Ej)
    end
    Hl = -Hl
    return Hr + Hl
end
@doc raw"""
Multiresolution Color Image Segmentation
Jianqing Liu and Yee-Hong Yang, Senior Member, IEEE

$F(I) = \sqrt{R} \times \sum\limits_{i=1}^R \dfrac{e_i^2}{\sqrt{A_i}}$
"""
struct LiuYangF
end

function scale_factor(c::LiuYangF, image::Matrix, segments::Matrix)
    return sqrt(maximum(segments))
end
@doc raw"""
´´´julia
function color_error_sum(image::Matrix, segments::Matrix{Integer)
´´´
Computes 
$\sum\limits_{i=1}^R \dfrac{e_i^2}{\sqrt{A_i}}$
where $R$ is the number of regions in the segmented image, $A_i$ is the area, or the number of pixels of the ith region $i$, and $e_i$  the color error of region $i$. e is defined as the sum of the Euclidean distance of the color vectors between the original image and the segmented image of each pixel in the region. 

"""
function color_error_sum(image::Matrix, segments::Matrix{Integer})
    labels = unique(segments)
    out = 0
    for i in labels
        pixels_in_segment = segments.==i
        A_i = count(pixels_in_segment)
        e_i = sum((norm.(mean(image[pixels_in_segment]) - image[pixels_in_segment])).^2)
        out += e_i/sqrt(A_i)
    end
    return (1/(prod(size(image))))*sqrt(scale_factor(c, image, segments))*out
end                         
function evaluate(c::LiuYangF, image::Matrix, segments::Matrix{Integer})
    return scale_factor(c, image, segments)*color_error_sum(image, segments)
end

"""
Quantitative evaluation of color image segmentation results 
M. Borsotti a, P. Campadelli a,2, R. Schettini b,
"""
struct FPrime
end

function scale_factor(c::FPrime, image::Matrix, segments::Matrix)
    segment_sizes = [count(segments.==i) for i in labels]
    scale = 0
    for A in unique(segments_sizes)
        R_A = count(segments_sizes.==seg_size)
        scale += R_A^(1 + (1/A))
    end
    return (1/prod(size(image))) *scale
end
function evaluate(c::FPrime, image::Matrix, segments::Matrix{T}) where T<:Integer
    return scale_factor(c, image, segments)*color_error_sum(image, segments)
end

@doc raw"""
Quantitative evaluation of color image segmentation results
M. Borsotti a, P. Campadelli a,2, R. Schettini b,

$Q(I) = \dfrac{1}{10000(N \times M)} \sqrt{R} \times \sum\limits_{i=1}^R \left[  \dfrac{e_i^2}{1+\log A_i} + \left(  \dfrac{R(A_i)}{A_i} \right)^2 \right]$

where $R$ is the number of regions in the segmented image, $A_i$ is the area, or the number of pixels of the ith region $i$, and $e_i$  the color error of region $i$. e is defined as the sum of the Euclidean distance of the color vectors between the original image and the segmented image of each pixel in the region, while $R(A_i)$ represents the number of regions having an area equal to $A_i$.
"""
struct Q
end
function evaluate(c::Q, image::Matrix, segments::Matrix{Integer})
    segment_sizes = [count(segments.==i) for i in labels]
    labels = unique(segments)
    out = 0
    for i in labels
        pixels_in_segment = segments.==i
        A_i = count(pixels_in_segment)
        e_i = sum((norm.(mean(image[pixels_in_segment]) - image[pixels_in_segment])).^2)
        R_A_i = count(segments_sizes.==A_i)
        out += e_i^2/(1 + log(A_i)) + (R_A_i/A_i)^2
    end
    return (1/(10000*prod(size(image))))*sqrt(maximum(labels))*out
end

"""
Performance Measures for Video Object Segmentation and Tracking
Erdem, Sankur, Tekalp
"""
struct ErdemMethod
    M::Integer # Neighborhood size to evaluate
    L::Integer # Length of the normal line
end
function mean_value(img::Matrix{T}, pos, w) where T
    R = CartesianRange(size(img))
    I, L = first(R), last(R)
    sum = zeros(T)
    range = CartesianRange(max(I, pos-w), min(L, pos+w))
    for N in  range
        sum += img[N]
    end
    return sum / prod(size(range))
end

function evaluate(c::ErdemMethod, image::Matrix, segments::Matrix{Integer})
    (h,w) = size(image)
    for i in unique(segments)
        segment = segments.==i
    end
    sum = 0
    (inside, outside) = normal_lines_extremes(boundary, c.L)
    for i=1:length(inside)
        val_inside = mean_value(image, inside[i,:], c.M)
        val_outside = mean_value(iamge, outside[i,:], c.M)
        delta_color = norm(val_outside - val_inside)/(sqrt(3*255^2))
        sum += delta_color
    end
    return 1 - (sum/length(inside))
end
