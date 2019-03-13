using Colors
using Statistics
using LinearAlgebra
using ImageSegmentation
using ColorTypes, ColorVectorSpace
export ECW,
    FRCRGBD,
    Zeboudj,
    ValuesEntropy,
    LiuYangF,
    FPrime,
    ErdemMethod,
    Q

import Base:
    -,
    iterate

(-)(a::Lab{T1}, b::Lab{T2}) where T1 where T2 =
    Lab{T1}(comp1(a)-comp1(b),
            comp2(a)-comp2(b),
            comp3(a)-comp3(b))

iterate(a::Lab) = (comp1(a), 1)
function iterate(a::Lab, i::Integer)
    if i == 1
        return (comp2(a), i+1)
    elseif i == 2
        return (comp3(a), i+1)
    else
        return nothing
    end
end
function evaluate(algo, img, segmented_image::SegmentedImage)
    return evaluate(algo, img, labels_map(segmented_image))
end


# On Selecting the Best Unsupervised Evaluation Techniques for Image Segmentation
"""
The use of visible color difference in the quantitative evaluation of color image segmentation,
Hsin-Chia Chen and Sheng-Jyh Wang 
"""
struct ECW
    threshold::Float64
    ECW(;threshold::Float64) = new(threshold)
end
function evaluate(c::ECW, image::AbstractArray{Color, N}, segments::SegmentedImage) where T<:Integer where {Color<:Lab, N}
    segments_mean = Dict(i=>convert(Lab, segment_mean(segments,i))
                     for i in segments.segment_labels)
    R = length(segments_mean)
    mean_segments = collect(map(
            i->segments_mean[i],
            segments.image_indexmap))
    E_intra = sum(norm.(image-mean_segments).<  c.threshold)
    C = 1/6
    total_sum = 0
    bool_map = [segments.image_indexmap .== i for i=1:R]
    for i=1:R-1
        a = bool_map[i]
        for j=i+1:R
            b = bool_map[j]         
            if (norm(segments_mean[i] - segments_mean[j])  > c.threshold)                
                 Kij = sum((a[1:end-1,1:end] + b[2:end,1:end]).==2)  + sum((a[1:end,1:end-1] + b[1:end,2:end]).==2)
                total_sum+=Kij                
            end
        end
    end
    E_inter= (2*total_sum)/(C*prod(size(segments.image_indexmap)))
    return 0.5*(E_intra +  E_inter)   
end

function evaluate(c::ECW, image::AbstractArray{Color, N}, segments::SegmentedImage) where {Color<:RGB, N}
    return evaluate(c, convert.(Lab, image), segments)
end


"""
Zéboudj, Rachid. Filtrage, seuillage automatique, contraste et contours: du pré-traitement à l'analyse d'image. Diss. Saint-Etienne, 1988.
Unsupervised Evaluation of Image Segmentation Application to Multi-spectral Images

"This contrast takes into account the internal and external contrast of the regions measured in the neighborhood of each pixel"
"""
struct Zeboudj
    radius::Integer
    Zeboudj(;radius::Integer=3) = new(radius)
end

function evaluate(c::Zeboudj, image::AbstractArray{Color, M}, segments::SegmentedImage) where {Color<:Colorant, M}
    image = convert.(Gray, image)
    N = maximum(segments.segment_labels)
    inside = zeros(N)
    outside = zeros(N)
    segment_sizes = zeros(N)
    border_lengths = zeros(N)
    R = CartesianIndices(segments.image_indexmap)
    I1, Iend = first(R), last(R)
    W = CartesianIndex(c.radius,c.radius)
    for I in CartesianIndices(size(image))
        current_label = segments.image_indexmap[I]
        segment_sizes[current_label] += 1
        max_inside = -1
        max_outside = -1
        is_a_border_point = false
        for J in max(I1, I-W):min(Iend, I+W)
            if current_label != segments.image_indexmap[J]
                is_a_border_point = true
                max_outside = max(max_outside, abs(image[I] - image[J]))
            else
                max_inside = max(max_inside, abs(image[I] - image[J]))
            end            
        end
        if (is_a_border_point)
            border_lengths[current_label]+=1
        end
        if max_inside == -1
            max_inside = 0
        end
        if max_outside == -1
            max_outside = 0
        end
        inside[current_label] += max_inside
        outside[current_label] += max_outside;
    end
    inside=inside ./ segment_sizes
    outside=outside ./ border_lengths

    C = zeros(N)
    for j=1:N
        if (inside[j] > 0 ) &&  (inside[j] < outside[j])
            C[j] = 1 - (inside[j]/outside[j])
        elseif (inside[j]==0)
            C[j] = outside[j]
        else
            C[j] = 0
        end
    end
    return (1/(prod(size(image)))) *sum(segment_sizes.*C)
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
function evaluate(c::ValuesEntropy, image::AbstractArray{C, N}, segments::SegmentedImage) where {C<:Colorant, N}
    Hr = 0
    Hl = 0 
    Si = prod(size(image))
    for j in segments.segment_labels
        segment = segments.image_indexmap .== j
        Sj = segments.segment_pixel_count[j]
        comps = components(image[segment])
        Hl += (Sj/Si)*(log(Sj/Si))
        H = 0
        for k=1:length(comps)
            for val in unique(comps[k])
                Lj = count(comps[k].==val)
                H += (Lj / Sj)*(log(Lj / Sj))
            end
        end
        Hr += (Sj/Si)*(-H)
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

function scale_factor(c::LiuYangF, image::AbstractArray{C,N}, segments::SegmentedImage) where {C<:Colorant, N}
    return sqrt(maximum(segments.segment_labels))
end
@doc raw"""
´´´julia
function color_error_sum(image::Matrix, segments::Matrix{Integer)
´´´
Computes 
$\sum\limits_{i=1}^R \dfrac{e_i^2}{\sqrt{A_i}}$
where $R$ is the number of regions in the segmented image, $A_i$ is the area, or the number of pixels of the ith region $i$, and $e_i$  the color error of region $i$. e is defined as the sum of the Euclidean distance of the color vectors between the original image and the segmented image of each pixel in the region. 

"""
function color_error_sum(c, image::AbstractArray{C, N}, segments::SegmentedImage) where {C<:Colorant, N}
    labels = segments.segment_labels
    out = 0
    for i in labels
        pixels_in_segment = segments.image_indexmap .== i
        A_i = segments.segment_pixel_count[i]
        e_i = sum((norm.(mean(image[pixels_in_segment]) - image[pixels_in_segment])).^2)
        out += e_i/sqrt(A_i)
    end
    return (1/(prod(size(image))))*sqrt(scale_factor(c, image, segments))*out
end                         
function evaluate(c::LiuYangF, image::AbstractArray{C, N}, segments::SegmentedImage) where {C<:Colorant, N}
    return scale_factor(c, image, segments)*color_error_sum(c, image, segments)
end

"""
Quantitative evaluation of color image segmentation results 
M. Borsotti a, P. Campadelli a,2, R. Schettini b,
"""
struct FPrime
end

function scale_factor(c::FPrime, image::AbstractArray{C, N}, segments::SegmentedImage) where {C<:Colorant, N}
    segment_sizes = values(segments.segment_pixel_count)
    scale = 0
    for A in unique(segment_sizes)
        R_A = count(segment_sizes .== A)
        scale += R_A^(1 + (1/A))
    end
    return (1/prod(size(image))) *scale
end
function evaluate(c::FPrime, image::AbstractArray{C, N}, segments::SegmentedImage)  where {C<:Colorant, N}
    return scale_factor(c, image, segments)*color_error_sum(c, image, segments)
end

@doc raw"""
Quantitative evaluation of color image segmentation results
M. Borsotti a, P. Campadelli a,2, R. Schettini b,

$Q(I) = \dfrac{1}{10000(N \times M)} \sqrt{R} \times \sum\limits_{i=1}^R \left[  \dfrac{e_i^2}{1+\log A_i} + \left(  \dfrac{R(A_i)}{A_i} \right)^2 \right]$

where $R$ is the number of regions in the segmented image, $A_i$ is the area, or the number of pixels of the ith region $i$, and $e_i$  the color error of region $i$. e is defined as the sum of the Euclidean distance of the color vectors between the original image and the segmented image of each pixel in the region, while $R(A_i)$ represents the number of regions having an area equal to $A_i$.
"""
struct Q
end
function evaluate(c::Q, image::AbstractArray{C, N}, segments::SegmentedImage) where {C<:Colorant, N}
    segment_sizes = collect(values(segments.segment_pixel_count))
    out = 0
    for i in segments.segment_labels
        pixels_in_segment = segments.image_indexmap .== i
        A_i = segments.segment_pixel_count[i]
        e_i = sum((norm.(mean(image[pixels_in_segment]) - image[pixels_in_segment])).^2)
        R_A_i = count(segment_sizes.==A_i)
        out += e_i^2/(1 + log(A_i)) + (R_A_i/A_i)^2
    end
    return (1/(10000*prod(size(image))))*sqrt(maximum(segments.segment_labels))*out
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
    R = CartesianIndices(size(img))
    I, L = first(R), last(R)
    sum = zeros(T)
    range = CartesianIndices(max(I, pos-w), min(L, pos+w))
    for N in  range
        sum += img[N]
    end
    return sum / prod(size(range))
end

function evaluate(c::ErdemMethod, image::AbstractArray{C, N}, segments::SegmentedImage) where {C<:Colorant, N}
    (h, w) = size(segments.image_indexmap)
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
                   
# RGB-D Images
struct FRCRGBD
end
function color_std(colors)
    return std(channelview(colors))
end

function evaluate(c::FRCRGBD,
                  color_image::AbstractArray{T},
                  range_image::Matrix,
                  seg_image::SegmentedImage) where T<:Colorant
    return evaluate(c,
                    convert.(Lab, color_image),
                    range_image,
                    labels_map(seg_image))
end
function evaluate(c::FRCRGBD,
                  color_image::Matrix{L},
                  range_image::Matrix,
                  segments::Matrix{<:Integer}) where L <: Lab
    color_image = channelview(color_image)
    N = size(segments, 1) * size(segments, 2)
    K = 0
    unique_segments = unique(segments)
    params = Dict()
    sigma_w = mapwindow(color_std,
                        colorview(RGB, color_image),
                        (3,3))

    for i in unique_segments
        mask = segments .== i
        indices = findall(mask)
        valsIi = color_image[:, indices]
        valsRi = range_image[indices]
        S_star = erode(mask)
        params[i] = Dict(
            :stdI=> std(valsIi),
            :stdR=> std(valsRi),
            :meanI=>dropdims(
                mean(valsIi, dims=[2, 3]),
                dims=2),
            :meanR=>mean(valsRi),
            :n=>sum(mask),
            :sigma_t=>sum(sigma_w[findall(S_star)])/sum(S_star), 
            :n_s_star=>sum(S_star) 
        )
    end
    DIntraI = 0
    DInterI = 0
    DIntraD = 0
    DInterD = 0
    for i in unique_segments
        if (params[i][:n_s_star] > 0)
            K = K + 1
            DIntraI += max(params[i][:stdI] - params[i][:sigma_t], 0)*(params[i][:n]/N)
            DIntraD += params[i][:stdR]*(params[i][:n]/N)
            for j in unique_segments
                if i != j
                    DInterI += norm(params[i][:meanI] - params[j][:meanI])
                    DInterD += norm(params[i][:meanR] - params[j][:meanR])
                end
            end
        else
            @warn("Segment to small")
        end
    end
    DInterI = DInterI / (K*(K-1) + eps())
    DInterD = DInterD / (K*(K-1) + eps())
    QColor = (DInterI - DIntraI) / 2 
    QDepth = (DInterD - DIntraD) / 2
    return QColor + 3*QDepth
end

