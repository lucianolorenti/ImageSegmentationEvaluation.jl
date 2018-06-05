
using Distances
using Clustering
include("csa.jl")
#include("SegmentationBenchmark.jl")
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
    BoundaryDisplacementError,
    thinning
using MLBase


struct FBoundary
    dmax::Float64
end

""""
```
function boundary_map(seg::Matrix{T}) where T<:Integer
```

% From a segmentation, compute a binary boundary map with 1 pixel wide
% boundaries.  The boundary pixels are offset by 1/2 pixel towards the
% origin from the actual segment boundary.
"""
function boundary_map(seg::Matrix{T}) where T<:Integer
    (h,w) = size(seg)

    local ee = zeros(size(seg));
    local s = zeros(size(seg));
    local se = zeros(size(seg));

    ee[:,1:end-1] = seg[ :,2:end];
    s[1:end-1,:] = seg[2:end,:];
    se[1:end-1,1:end-1] = seg[2:end,2:end];

    local b = (seg.!=ee) .| (seg.!=s) .| (seg.!=se);
    b[end,:] = seg[end,:].!=ee[end,:];
    b[:,end] = seg[:,end].!=s[:,end];
    b[end,end] = 0;
    return b
end
using MatrixNetworks
function pixel_mapping(cl::AbstractMatrix, gt::AbstractMatrix, dmax)
    local pixToIdxCL = Dict{Tuple{Integer,Integer},Integer}()
    local pixToIdxGT = Dict{Tuple{Integer,Integer},Integer}()
    local idxToPixCL = Vector{Tuple{Integer,Integer}}()
    local idxToPixGT = Vector{Tuple{Integer,Integer}}()
    local ee         = Vector() 
    local matchableGT = falses(size(cl))
    local matchableCL = falses(size(cl))
    local nCL = 0
    local nGT = 0
    local r = round(Integer,ceil(dmax))
    for j in find(cl)
        (rc,cc) = ind2sub(size(cl),j)
        local wInit   = CartesianIndex(max.((rc,cc).-(r,r),(1,1)))        
        local wEnd    = CartesianIndex(min.((rc,cc).+(r,r),size(cl)))
        for CI in CartesianRange( wInit, wEnd)
            if gt[CI]
                local k =  sub2ind(size(gt), CI[1],CI[2])
                local dist = sqrt((CI[1]-rc)^2 +  (CI[2]-cc)^2)
                if dist <= dmax
                    matchableCL[rc,cc] = true
                    matchableGT[CI] = true
                    push!(ee, ( ( (rc,cc),  (CI[1], CI[2])), round(Integer,dist*100)))
                end
            end
        end
    end
    for J in CartesianRange(size(cl))
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
function evaluate(f::FBoundary, cl_seg::Matrix{T}, gt_seg::Matrix{T}) where T<:Integer

    local cl = thinning(boundary_map(cl_seg))
    local gt = thinning(boundary_map(gt_seg))
    local scale_cost =  sqrt( size(cl,1)^2 + size(cl,2)^2)
    local dmax = f.dmax * scale_cost
    local noutliers = 6
    local cl_detected = find(cl)
    local gt_detected = find(gt)
    local outlier_weight =-100*f.dmax*scale_cost

    (pixToIdxCL, idxToPixCL, nCL, matchableCL, pixToIdxGT, idxToPixGT, nGT, matchableGT, ee) = pixel_mapping(cl, gt, dmax)

    local I = sizehint!(Vector{Int64}(),(nCL+nGT)*6)
    local J = sizehint!(Vector{Int64}(),(nCL+nGT)*6)
    local w     = sizehint!(Vector{Float64}(),(nCL+nGT)*6)
    
    Aidx(x) = Integer(x)
    Aoutlier(x) = Integer(nCL+x)
    Bidx(x) = Integer(nCL+nGT+x)
    Boutlier(x) = Integer(nCL+nGT+nGT+x)

    
    local n = 2*(nCL+nGT)
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
        
    local match = bipartite_matching(sparse(I,J,w,n,n))
    (cl_edge, gt_edge) = edge_list(match)
    local matchCL = zeros(Integer,size(cl))
    local matchG = zeros(Integer,size(cl))
    local ff = 0
    for j=1:length(cl_edge)
        local v1 = cl_edge[j]
        local v2 = gt_edge[j] - (nCL+nGT)
        if (v1 > nCL+nGT)
            local t = v1
            v1 = v2
            v2 = t
        end
        if (v1<= nCL) &&  (v2<=nGT)
            local pix1 = idxToPixCL[v1]
            if cl[pix1[1],pix1[2]]                
                matchCL[pix1[1], pix1[2]] = v2
            end
            
            local pix2 = idxToPixGT[v2]
            if gt[pix2[1],pix2[2]]
                
                matchG[pix2[1],pix2[2]] = v1
            end
        end
    end

    local cntR = sum(matchG.>0)
    local sumR = length(gt_detected)
    local cntP = sum(matchCL.>0)
    local sumP = length(cl_detected)
    local rec = cntR/sumR;
    local prec = cntP/sumP;
    return ((2*prec*rec)/(prec+rec), prec, rec)
end

function evaluate(d, cl::Matrix{T}, gt::Matrix{T}) where T<:Integer
    local N = prod(size(cl))
    local clusters_cl = [(cl.==j) for j in unique(cl)]
    local clusters_gt = [(gt.==j) for j in unique(gt)]

    local sum_tot = 0
    for c in clusters_cl
        local length = sum(c)
        sum_tot = sum_tot + maximum([length*edge_weight(d, c,g) for g in clusters_gt])            
    end
    return sum_tot / N
    
end
type Precision end
function edge_weight(d::Precision, cl::BitArray{2}, gt::BitArray{2})
    return precision(roc(vec(gt),vec(cl)))
end
type FMeasure end
function edge_weight(d::FMeasure, cl::BitArray{2}, gt::BitArray{2})
    return f1score(roc(vec(gt),vec(cl)))
end
type SegmentationCovering end
function edge_weight(d::SegmentationCovering, cl::BitArray{2}, gt::BitArray{2})
    return 1-jaccard(Integer.(vec(gt)),Integer.(vec(cl)))
end
type VariationOfInformation
    normalize::Bool
end
function VariationOfInformation()
    return VariationOfInformation(true)
end  

function H(clusters::Vector{BitArray{2}})    
    local n = prod(size(clusters[1]))    
    local tot_sum = 0
    for c in clusters
        local R = sum(c)
        tot_sum = tot_sum + R*log(R/n)
    end
    return (-1/n)*tot_sum    
end
function I(clusters1::Vector{BitArray{2}}, clusters2::Vector{BitArray{2}})
    local tot_sum = 0
    local n = prod(size(clusters1[1]))
    for c1 in clusters1
        local R1 = sum(c1)
        for c2 in clusters2
            local R2 = sum(c2)
            local intersection = sum(c1 .& c2)
            if (intersection>0)
                tot_sum = tot_sum + intersection *log(((n*intersection)/(R1*R2)))
            end
        end
    end
    return (1/n)*tot_sum
            
end
function evaluate(d::VariationOfInformation, cl::Matrix{T}, gt::Matrix{T}) where T<:Integer
    local ff = varinfo(maximum(cl), vec(cl),maximum(gt),vec(gt))
    local clusters_cl = [(cl.==j) for j in unique(cl)]
    local clusters_gt = [(gt.==j) for j in unique(gt)]
    local voi =  H(clusters_cl) + H(clusters_gt) - 2*I(clusters_cl,clusters_gt)
    if (d.normalize)
        return voi / 2*log(max(maximum(cl),maximum(gt)))
    else
        return voi
    end
end
struct RandIndex end
function evaluate(d::RandIndex, c1::Matrix{T}, gt::Matrix{T}) where T<:Integer
    local n = prod(size(c1))
    local M = zeros(2,2)
    (a,b,c,d) = randindex(vec(c1), vec(gt))
    return b
end
struct FMeasureRegions end

doc""""
```julia
function relabel(c1::Matrix}, part_bimap=Dict{Integer,Integer}()) where T<:Integer
```
It relabels a partition in scanning order. Bimaps are the look up tables of the relabeling.
  - author Jordi Pont Tuset <jordi.pont@upc.edu>
"""
function relabel(c1::Matrix{T}, bimap=Dict{Integer,Integer}()) where T<:Integer
    local partition_out = zeros(Integer, size(c1))
    local max_region = 1;
    for I in CartesianRange(size(c1))
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
doc"""
```julia
function evaluate(d::FMeasureRegions, c1::Matrix{T}, gt::Matrix{T}) where T<:Integer
```

# References
- https://github.com/jponttuset/seism/blob/c0f10b4509d364e39e5dba28fe57c2d157c0808e/src/misc/intersection_matrix.hpp
- https://github.com/JuliaStats/Clustering.jl/blob/master/src/randindex.jl
"""

function evaluate(d::FMeasureRegions, c1::Matrix{T}, gt::Matrix{T}) where T<:Integer

    assert(size(c1)==size(gt))

    (partition1_relab, num_reg_1) = relabel(c1)
    (partition2_relab, num_reg_2) = relabel(gt)
    local c = Clustering.counts(partition1_relab, partition2_relab,(1:maximum(partition1_relab),1:maximum(partition2_relab))) # form contingency matrix

    local n = round(Int,sum(c))
    local nis = sum(sum(c,2).^2)        # sum of squares of sums of rows
    local njs = sum(sum(c,1).^2)        # sum of squares of sums of columns
    local t2 = sum(c.^2)                # sum over rows & columnns of nij^2
    local t3 = .5*(nis+njs)
    local n11 = (t2-n)/2
    local n00 = (n^2 - nis -njs +t2)/2
    local n10 = (nis-t2)/2
    local n01 = (njs-t2)/2

    local precision = n11/(n11+n10);
    local recall    = n11/(n11+n01);
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
end


function calculate_regions(num_reg_gt::Integer, num_reg_part::Integer,  intersect_matrix::Matrix)
    local image_area        = 0;    
    local region_areas_gt   = zeros(Integer,num_reg_gt)
    local region_areas_part = zeros(Integer, num_reg_part);    
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
    
    local area_percentile = 0.99;
    local area_map  = Vector{Tuple{Float64, Integer}}() # Mapping between each region area and its id
    local candidates = Dict{Integer,Bool}()
    #Get candidates in the partition (remove percentile of small area)
    for ii=1:length(region_areas)
        push!(area_map,((region_areas[ii])/float(image_area),ii));
    end
    sort!(area_map, by=x->x[1],rev=true)
    local curr_pct = 0;
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
function evaluate(c::PRObjectsAndParts, assignments_seg::Matrix{T}, assignments_gt::Matrix{T}) where T<:Integer

    const NOT_CLASSIFIED = 1
    const OBJECT         = 2
    const PART           = 3

    assignments_seg = relabel(assignments_seg)[1]
    assignments_gt  = relabel(assignments_gt)[1]
    local intersect_matrix = Clustering.counts(assignments_seg, assignments_gt,(1:maximum(assignments_seg),1:maximum(assignments_gt))) # form contingency matrix
   
    local num_reg_part        = size(intersect_matrix,1)
    local classification_part = ones(Integer,num_reg_part)
    local prec_part           = zeros(num_reg_part);
    local mapping_gt          = Dict{Integer,Integer}()
    local mapping_part        = Dict{Integer,Integer}()
    local num_reg_gt          = size(intersect_matrix,2)
    local classification_gt   = ones(Integer, num_reg_gt)
    local recall_gt           = zeros(num_reg_gt);
    
    (region_areas_gt, region_areas_part, image_area) =  calculate_regions(num_reg_gt, num_reg_part, intersect_matrix)

    local candidate_part = get_candidates(image_area, region_areas_part)
    local candidate_gt   = get_candidates(image_area, region_areas_gt)

    # Scan through table and find all OBJECT mappings */
    for ii=1:num_reg_gt
        for jj=1:num_reg_part
            local recall    = intersect_matrix[jj,ii]/float(region_areas_gt[ii]);
            local precision = intersect_matrix[jj,ii]/float(region_areas_part[jj]);
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
    local num_objects_part = 0;
    local num_objects_gt = 0;
    local num_parts_part = 0;
    local num_parts_gt = 0;
    local num_underseg_part = 0;
    local num_overseg_gt = 0;
    local num_candidates_part = 0;
    local num_candidates_gt = 0;

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
    local precision = (num_objects_part + num_underseg_part + c.B*num_parts_part)/float(num_candidates_part);
    local recall    = (num_objects_gt   + num_overseg_gt    + c.B*num_parts_gt  )/float(num_candidates_gt);

    # F-measure for Region Detection
    if(precision==0 && recall==0)
        f_measure = (0,0,0) ;
    else
        f_measure = (2*precision*recall/(precision+recall), precision, recall)
    end
end



"""
Authors: John Wright, and Allen Y. Yang
Contact: Allen Y. Yang <yang@eecs.berkeley.edu>
"""
type BoundaryDisplacementError
end
using Images
function evaluate(cfg::BoundaryDisplacementError, cl::Matrix{T}, gt::Matrix{T})  where T<:Integer
    assert(size(cl)==size(gt))

    
    # Generate boundary maps
    (gradX, gradY)=imgradients(cl, KernelFactors.ando3)
    (boundaryPixelY, boundaryPixelX) =ind2sub(size(cl),find((abs.(gradX)+abs.(gradY)).!=0)) 
    local boundary1 = (abs.(gradX) .+ abs.(gradY)) .> 0;

    
    (gradX, gradY)=imgradients(gt, KernelFactors.ando3)
    (boundaryPixelY, boundaryPixelX) =ind2sub(size(cl),find((abs.(gradX)+abs.(gradY)).!=0)) 
    local boundary2 = (abs.(gradX) .+ abs.(gradY)) .> 0;


    # boundary1 and boundary2 are now binary boundary masks. compute their distance transforms:
    local D1 = distance_transform(feature_transform(boundary1));
    local D2 = distance_transform(feature_transform(boundary2));

    # compute the distance of the pixels in boundary1 to the nearest pixel in% boundary2:
    local dist_12 = sum(boundary1 .* D2 );
    local dist_21 = sum(boundary2 .* D1 );

    local avgError_12 = dist_12 / sum(boundary1);
    local avgError_21 = dist_21 / sum(boundary2);
    return (avgError_12 + avgError_21) / 2;
    
end
