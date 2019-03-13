
using Distances
using Clustering
using MatrixNetworks
using ImageSegmentation
using SparseArrays
export
    edge_weight,
    boundary_map,
    Precision,
    FMeasure,
    FBoundary,
    SegmentationCovering,
    VariationOfInformation,
    RandIndex,
    FMeasureRegions,
    PRObjectsAndParts,
    BoundaryDisplacementError
using MLBase



function evaluate(f, cl_seg::SegmentedImage, gt_seg::SegmentedImage)
    return evaluate(f, labels_map(cl_seg), labels_map(gt_seg))
end
function evaluate(f, cl_seg::Matrix{T1}, gt_seg::Matrix{T2}) where T1<:Integer where T2<:Integer
    return evaluate(f, convert.(Integer, cl_seg), convert.(Integer, gt_seg))
end

"""
```
struct FBoundary
```

Learning to Detect Natural Image Boundaries Using Local Brightness, Color, and Texture Cues
David R. Martin, Member, IEEE, Charless C. Fowlkes, and Jitendra Malik, Member, IEEE

# Members
- `dmax::Float64`
"""
struct FBoundary
    dmax::Float64
    FBoundary(;dmax::Float64=0.0075) = new(dmax)
end

function pixel_mapping(cl::AbstractMatrix, gt::AbstractMatrix, dmax)
    pixToIdxCL = Dict{Tuple{Integer,Integer},Integer}()
    pixToIdxGT = Dict{Tuple{Integer,Integer},Integer}()
    idxToPixCL = Vector{Tuple{Integer,Integer}}()
    idxToPixGT = Vector{Tuple{Integer,Integer}}()
    ee = Vector() 
    matchableGT = falses(size(cl))
    matchableCL = falses(size(cl))
    nCL = 0
    nGT = 0
    r = round(Integer,ceil(dmax))
    for j in findall(cl)
        (rc,cc) = Tuple(CartesianIndices(size(cl))[j])
        row_range = max(rc-r,1):min(rc+r, size(cl,1))
        col_range = max(cc-r,1):min(cc+r,size(cl,2))
        for CI in CartesianIndices((row_range, col_range))
            if gt[CI]
                k =  LinearIndices(size(gt))[CI[1],CI[2]]
                dist = sqrt((CI[1]-rc)^2 +  (CI[2]-cc)^2)
                if dist <= dmax
                    matchableCL[rc,cc] = true
                    matchableGT[CI] = true
                    push!(ee, ( ( (rc,cc),  (CI[1], CI[2])), round(Integer,dist*100)))
                end
            end
        end
    end
    for J in CartesianIndices(size(cl))
        if (matchableCL[J])            
            nCL = nCL +1
            pixToIdxCL[(J[1],J[2])] = nCL
            push!(idxToPixCL, (J[1],J[2]))
        end
        if matchableGT[J]
            nGT = nGT +1
            pixToIdxGT[(J[1],J[2])] = nGT
            push!(idxToPixGT, (J[1], J[2]))
        end
    end
    return (pixToIdxCL, idxToPixCL, nCL, matchableCL, pixToIdxGT, idxToPixGT, nGT, matchableGT, ee)
end
function evaluate(f::FBoundary, cl_seg::Matrix{T1}, gt_seg::Matrix{T2}) where T1<:Integer where T2<:Integer

     cl = thinning(boundary_map(BoundaryGradient(), cl_seg))
     gt = thinning(boundary_map(BoundaryGradient(), gt_seg))
     scale_cost =  sqrt( size(cl,1)^2 + size(cl,2)^2)
     dmax = f.dmax * scale_cost
     noutliers = 6
     cl_detected = findall(cl)
     gt_detected = findall(gt)
     outlier_weight =-100*f.dmax*scale_cost

    (pixToIdxCL, idxToPixCL, nCL, matchableCL, pixToIdxGT, idxToPixGT, nGT, matchableGT, ee) = pixel_mapping(cl, gt, dmax)

     I = sizehint!(Vector{Int64}(),(nCL+nGT)*6)
     J = sizehint!(Vector{Int64}(),(nCL+nGT)*6)
     w     = sizehint!(Vector{Float64}(),(nCL+nGT)*6)
    
    Aidx(x) = Integer(x)
    Aoutlier(x) = Integer(nCL+x)
    Bidx(x) = Integer(nCL+nGT+x)
    Boutlier(x) = Integer(nCL+nGT+nGT+x)

    
     n = 2*(nCL+nGT)
    for (edge, weight) in ee
        push!(I, Aidx(pixToIdxCL[edge[1]]))
        push!(J, Bidx(pixToIdxGT[edge[2]]))
        push!(w    , weight) 
        
    end
    for i=1:nCL
        for j in rand(1:nCL-1, noutliers)
            if (i==j)
                j=j+1
            end
            
            push!(I,Aidx(i))
            push!(J, Boutlier(j))
            push!(w, outlier_weight)
        end
    end
    
    for i=1:nGT
        for j in rand(1:nGT-1, noutliers)
            if (i==j)
                j=j+1
            end
            
            push!(I,Aoutlier(j))
            push!(J,Bidx(i))
            push!(w, outlier_weight)
        end
    end
    for i=1:max(nGT,nCL)
        for j in rand(1:min(nGT,nCL), noutliers)
            if (nGT<nCL)
                
                push!(I, Aoutlier(j))
                push!(J,Boutlier(i))
            else
                push!(I, Aoutlier(i))
                push!(J, Boutlier(j))
            end
            push!(w, outlier_weight)
        end
    end
   
    for i=1:nCL
        push!(I,Aidx(i))
        push!(J,Boutlier(i))
        push!(w, outlier_weight)
    end
    
    for i=1:nGT
        push!(I, Aoutlier(i))
        push!(J,Bidx(i))
        push!(w, outlier_weight)
    end
        
     match = bipartite_matching(sparse(I,J,w,n,n))
    (cl_edge, gt_edge) = edge_list(match)
     matchCL = zeros(Integer,size(cl))
     matchG = zeros(Integer,size(cl))
     ff = 0
    for j=1:length(cl_edge)
         v1 = cl_edge[j]
         v2 = gt_edge[j] - (nCL+nGT)
        if (v1 > nCL+nGT)
             t = v1
            v1 = v2
            v2 = t
        end
        if (v1<= nCL) &&  (v2<=nGT)
             pix1 = idxToPixCL[v1]
            if cl[pix1[1],pix1[2]]                
                matchCL[pix1[1], pix1[2]] = v2
            end
            
             pix2 = idxToPixGT[v2]
            if gt[pix2[1],pix2[2]]
                
                matchG[pix2[1],pix2[2]] = v1
            end
        end
    end
    
    cntR = sum(matchG.>0)
    sumR = length(gt_detected)
    cntP = sum(matchCL.>0)
    sumP = length(cl_detected)
    rec = cntR/(sumR +  eps());
    prec = cntP/(sumP + eps());
    if sumR == 0
        @warn("Infinit recall")
    end
    if sumP == 0
        @warn("Infinit precision")
    end
    return ((2*prec*rec)/(prec+rec + eps()), prec, rec)
end

function evaluate(d, cl::Matrix{T}, gt::Matrix{T}) where T<:Integer
     N = prod(size(cl))
     clusters_cl = [(cl.==j) for j in unique(cl)]
     clusters_gt = [(gt.==j) for j in unique(gt)]

     sum_tot = 0
    for c in clusters_cl
         length = sum(c)
        sum_tot = sum_tot + maximum([length*edge_weight(d, c,g) for g in clusters_gt])            
    end
    return sum_tot / N
    
end
"""
Precision
"""
struct Precision
end
function edge_weight(d::Precision, cl::BitArray{2}, gt::BitArray{2})
    return precision(roc(vec(gt),vec(cl)))
end
"""
FMeasure
"""
struct FMeasure
end
function edge_weight(d::FMeasure, cl::BitArray{2}, gt::BitArray{2})
    return f1score(roc(vec(gt),vec(cl)))
end
"""
Segmentation Covering
"""
struct SegmentationCovering end
function edge_weight(d::SegmentationCovering, cl::BitArray{2}, gt::BitArray{2})
    return 1-jaccard(Integer.(vec(gt)),Integer.(vec(cl)))
end
"""
Variation of Information
"""
struct VariationOfInformation
    normalize::Bool
end
function VariationOfInformation()
    return VariationOfInformation(true)
end  

function H(clusters::Vector{BitArray{2}})    
     n = prod(size(clusters[1]))    
     tot_sum = 0
    for c in clusters
         R = sum(c)
        tot_sum = tot_sum + R*log(R/n)
    end
    return (-1/n)*tot_sum    
end
function I(clusters1::Vector{BitArray{2}}, clusters2::Vector{BitArray{2}})
     tot_sum = 0
     n = prod(size(clusters1[1]))
    for c1 in clusters1
         R1 = sum(c1)
        for c2 in clusters2
             R2 = sum(c2)
             intersection = sum(c1 .& c2)
            if (intersection>0)
                tot_sum = tot_sum + intersection *log(((n*intersection)/(R1*R2)))
            end
        end
    end
    return (1/n)*tot_sum
            
end
function evaluate(d::VariationOfInformation, cl::Matrix{T}, gt::Matrix{T}) where T<:Integer
    ff = varinfo(maximum(cl), vec(cl), maximum(gt),vec(gt))
    clusters_cl = [(cl.==j) for j in unique(cl)]
    clusters_gt = [(gt.==j) for j in unique(gt)]
    voi =  H(clusters_cl) + H(clusters_gt) - 2*I(clusters_cl,clusters_gt)
    if (d.normalize)
        return voi / 2*log(max(maximum(cl),maximum(gt)))
    else
        return voi
    end
end
"""
RandIndex
"""
struct RandIndex end
function evaluate(d::RandIndex, c1::Matrix{T}, gt::Matrix{T}) where T<:Integer
     n = prod(size(c1))
     M = zeros(2,2)
    (a,b,c,d) = randindex(vec(c1), vec(gt))
    return b
end
"""
FMeasure Regions
"""
struct FMeasureRegions end

""""
```julia
function relabel(c1::Matrix}, part_bimap=Dict{Integer,Integer}()) where T<:Integer
```
It relabels a partition in scanning order. Bimaps are the look up tables of the relabeling.
  - author Jordi Pont Tuset <jordi.pont@upc.edu>
"""
function relabel(c1::Matrix{T}, bimap=Dict{Integer,Integer}()) where T<:Integer
     partition_out = zeros(Integer, size(c1))
     max_region = 1;
    for I in CartesianIndices(size(c1))
        if !haskey(bimap, c1[I])
            bimap[c1[I]] = max_region;
            partition_out[I] = max_region;
            max_region=max_region+1;
        else
            partition_out[I] = bimap[c1[I]]
        end
    end
    return (partition_out, max_region-1)
end
"""
```julia
function evaluate(d::FMeasureRegions, c1::Matrix{T}, gt::Matrix{T}) where T<:Integer
```

# References
- https://github.com/jponttuset/seism/blob/c0f10b4509d364e39e5dba28fe57c2d157c0808e/src/misc/intersection_matrix.hpp
- https://github.com/JuliaStats/Clustering.jl/blob/master/src/randindex.jl
"""

function evaluate(d::FMeasureRegions, c1::Matrix{T}, gt::Matrix{T}) where T<:Integer

    @assert(size(c1)==size(gt))

    (partition1_relab, num_reg_1) = relabel(c1)
    (partition2_relab, num_reg_2) = relabel(gt)
    c = Clustering.counts(partition1_relab,
                          partition2_relab,
                          (1:maximum(partition1_relab),
                           1:maximum(partition2_relab))) # form contingency matrix
    n = round(Int,sum(c))
    nis = sum(sum(c, dims=2).^2)        # sum of squares of sums of rows
    njs = sum(sum(c, dims=1).^2)        # sum of squares of sums of columns
    t2 = sum(c.^2)                # sum over rows & columnns of nij^2
    t3 = .5*(nis+njs)
    n11 = (t2-n)/2
    n00 = (n^2 - nis -njs +t2)/2
    n10 = (nis-t2)/2
    n01 = (njs-t2)/2
    
    precision = n11/(n11+n10);
    recall    = n11/(n11+n01);
    if (precision+recall>0)
        return (2*precision*recall/(precision+recall), precision, recall)
    else
        return (0,0,0)
    end
end

"""

http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Pont-Tuset_Measures_and_Meta-Measures_2013_CVPR_paper.pdf

Measures and Meta-Measures for the Supervised Evaluation of Image Segmentation Jordi Pont-Tuset and Ferran Marques. Universitat Politecnica de Catalunya BarcelonaTech

"""
struct PRObjectsAndParts 
    object_threshold::Float64
    part_threshold::Float64
    B::Float64
    PRObjectsAndParts(;
                      object_threshold::Float64=0.95,
                      part_threshold::Float64=0.25,
                      B::Float64=0.1) = new(object_threshold,
                                            part_threshold,
                                            B)
end


function calculate_regions(num_reg_gt::Integer, num_reg_part::Integer,  intersect_matrix::Matrix)
     image_area        = 0;    
     region_areas_gt   = zeros(Integer,num_reg_gt)
     region_areas_part = zeros(Integer, num_reg_part);    
    for ii=1:num_reg_gt
        for jj=1:num_reg_part
            image_area += intersect_matrix[jj,ii];
            region_areas_part[jj] = region_areas_part[jj] + intersect_matrix[jj,ii];
            region_areas_gt[ii]   = region_areas_gt[ii]   + intersect_matrix[jj,ii];
        end
    end
    return (region_areas_gt, region_areas_part, image_area)
end
function get_candidates(image_area::Integer, region_areas::Vector)
    
     area_percentile = 0.99;
     area_map  = Vector{Tuple{Float64, Integer}}() # Mapping between each region area and its id
     candidates = Dict{Integer,Bool}()
    #Get candidates in the partition (remove percentile of small area)
    for ii=1:length(region_areas)
        push!(area_map,((region_areas[ii])/float(image_area),ii));
    end
    sort!(area_map, by=x->x[1],rev=true)
     curr_pct = 0;
    for (area_prop, area) in area_map
        if (curr_pct < area_percentile)
            candidates[area] = true
        else
            candidates[area] = false
        end
        curr_pct += area_prop
    end
    return candidates
end
function evaluate(c::PRObjectsAndParts,
                  assignments_seg::Matrix{T},
                  assignments_gt::Matrix{T}) where T<:Integer

    NOT_CLASSIFIED = 1
    OBJECT         = 2
    PART           = 3

    assignments_seg = relabel(assignments_seg)[1]
    assignments_gt  = relabel(assignments_gt)[1]
     intersect_matrix = Clustering.counts(assignments_seg, assignments_gt,(1:maximum(assignments_seg),1:maximum(assignments_gt))) # form contingency matrix
   
     num_reg_part        = size(intersect_matrix,1)
     classification_part = ones(Integer,num_reg_part)
     prec_part           = zeros(num_reg_part);
     mapping_gt          = Dict{Integer,Integer}()
     mapping_part        = Dict{Integer,Integer}()
     num_reg_gt          = size(intersect_matrix,2)
     classification_gt   = ones(Integer, num_reg_gt)
     recall_gt           = zeros(num_reg_gt);
    
    (region_areas_gt, region_areas_part, image_area) =  calculate_regions(num_reg_gt, num_reg_part, intersect_matrix)

     candidate_part = get_candidates(image_area, region_areas_part)
     candidate_gt   = get_candidates(image_area, region_areas_gt)

    # Scan through table and find all OBJECT mappings */
    for ii=1:num_reg_gt
        for jj=1:num_reg_part
             recall    = intersect_matrix[jj,ii]/float(region_areas_gt[ii]);
             precision = intersect_matrix[jj,ii]/float(region_areas_part[jj]);
            # Ignore those regions with tiny area */
            if(candidate_gt[ii]==true && candidate_part[jj]==true)
                # Is it an object candidate? */
                if(recall >= c.object_threshold)  &&  (precision >= c.object_threshold)
                    classification_gt[ii]  = OBJECT;
                    classification_part[jj] = OBJECT;
                    mapping_gt[ii] = jj;
                    mapping_part[jj]        = ii;
                elseif (recall >= c.part_threshold)  &&  (precision >= c.object_threshold)
                    if (classification_part[jj] == NOT_CLASSIFIED)
                        classification_part[jj] = PART;
                        mapping_part[jj]        = ii;
                    end
                elseif(recall >= c.object_threshold)  &&  (precision >= c.part_threshold)
                        # Cannot have a classification already */
                        classification_gt[ii] = PART;
                        mapping_gt[ii] = jj;
                end               
            end
            
            #Get _recall_gt and _prec_part (no matter if candidates or not), discarding objects
            if(precision >= c.object_threshold) && (recall < c.object_threshold)
                recall_gt[ii] += recall;
            elseif (recall >= c.object_threshold) && (precision < c.object_threshold)
                prec_part[jj] += precision;
            end
        end
    end
    

    #Count everything
     num_objects_part = 0;
     num_objects_gt = 0;
     num_parts_part = 0;
     num_parts_gt = 0;
     num_underseg_part = 0;
     num_overseg_gt = 0;
     num_candidates_part = 0;
     num_candidates_gt = 0;

    for jj=1:num_reg_part
        num_candidates_part += candidate_part[jj];

        if (classification_part[jj]==PART)
            num_parts_part = num_parts_part+1;
        elseif(classification_part[jj]==OBJECT)
            num_objects_part = num_objects_part +1 ;
        elseif (candidate_part[jj]) # Compute degree of undersegmentation
            num_underseg_part += prec_part[jj];
        end
    end
    for ii=1:num_reg_gt
        num_candidates_gt += candidate_gt[ii];
        if (classification_gt[ii]==PART)
                num_parts_gt = num_parts_gt+1;
        elseif(classification_gt[ii]==OBJECT)
                num_objects_gt = num_objects_gt+1;
        elseif(candidate_gt[ii])
                num_overseg_gt =  num_overseg_gt + recall_gt[ii];
        end
    end
    # Precision and recall
     precision = (num_objects_part + num_underseg_part + c.B*num_parts_part)/float(num_candidates_part);
     recall    = (num_objects_gt   + num_overseg_gt    + c.B*num_parts_gt  )/float(num_candidates_gt);

    # F-measure for Region Detection
    if(precision==0 && recall==0)
        f_measure = (0,0,0) ;
    else
        f_measure = (2*precision*recall/(precision+recall), precision, recall)
    end
end


"""
Boundary Displacement Error
"""
struct BoundaryDisplacementError
end
using Images
function evaluate(cfg::BoundaryDisplacementError, cl::Matrix{T}, gt::Matrix{T})  where T<:Integer
    @assert(size(cl)==size(gt))
    # Generate boundary maps
    boundary1 = boundary_map(BoundaryGradient(), cl)
    boundary2 = boundary_map(BoundaryGradient(), gt)
    
    # boundary1 and boundary2 are now binary boundary masks. compute their distance transforms:
    D1 = distance_transform(feature_transform(boundary1));
    D2 = distance_transform(feature_transform(boundary2));

    # compute the distance of the pixels in boundary1 to the nearest pixel in% boundary2:
    dist_12 = sum(boundary1 .* D2 );
    dist_21 = sum(boundary2 .* D1 );
    avgError_12 = dist_12 / sum(boundary1);
    avgError_21 = dist_21 / sum(boundary2);
    return (avgError_12 + avgError_21) / 2;
end
