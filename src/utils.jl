"""
```
function LineNormals2D(vertices)
```
This function calculates the normals, of the line points
using the neighbouring points of each contour point, and
forward an backward differences on the end points
"""
function LineNormals2D(vertices)
    n = size(vertices,1)
    # Calculate tangent vectors
    DT = float.(vertices[1:n-1,:] .- vertices[2:n,:])

    # Make influence of tangent vector 1/Distance
    # (Weighted Central Differences. Points which are closer give a
    # more accurate estimate of the normal)
    LL=sqrt.(DT[ :,1].^2 .+ DT[:,2].^2);
    DT ./= max.(LL.^2, eps())
    
    D = vcat([0 0], DT) + vcat(DT, [0 0])
    #D=D1+D2;
    # Normalize the normal
    LL=sqrt.(D[:,1].^2+D[:,2].^2);
    N = zeros(size(vertices))
    N[:,1] = -D[:,2] ./ LL;
    N[:,2] = D[:, 1]./ LL;

    return N
end

struct BoundaryGradient
end

function boundary_map(b::BoundaryGradient, img::Matrix)
    (nr, nc) = size(img)
    gradX = zeros(size(img))
    for c = 2:nc-1
        gradX[:,c] = 0.5*(img[:,c+1] - img[:,c-1]);
    end
    gradX[:,1] = img[:,2] - img[:,1];
    gradX[:,nc] = img[:,nc] - img[:,nc-1];
    gradY = zeros(size(img))
    for r = 2:nr-1
        gradY[r,:] = 0.5*(img[r+1,:] - img[r-1,:]);
    end
    gradY[1,:] = img[2,:] - img[1,:];
    gradY[nr,:] = img[nr,:] - img[nr-1,:];
    return (abs.(gradX) .+ abs.(gradY)) .> 0;
end

struct BoundaryShift
end

""""
```
function boundary_map(seg::Matrix{T}) where T<:Integer
```
From a segmentation, compute a binary boundary map with 1 pixel wide  boundaries. 
The boundary pixels are offset by 1/2 pixel towards the origin from the actual segment boundary.
"""
function boundary_map(cfg::BoundaryShift, seg::Matrix{T}) where T<:Integer
    (h,w) = size(seg)

    local ee = zeros(size(seg));
    local s = zeros(size(seg));
    local se = zeros(size(seg));

    ee[:,1:end-1] = seg[ :,2:end];
    s[1:end-1,:] = seg[2:end,:];
    se[1:end-1,1:end-1] = seg[2:end,2:end];

    local b = (seg.!=ee) .| (seg.!=s) .| (seg.!=se);
    b[end,:] = seg[end,:] .!= ee[end,:];
    b[:,end] = seg[:,end] .!= s[:,end];
    b[end,end] = 0;
    return b
end


struct BoundaryMooreTracing
end
function boundary_map(cfg::BoundaryMooreTracing, binary::Matrix)

    # Pad the input image with a 1-px border.
    ( rows, columns ) = size( binary );
    padded = falses(rows +2, columns +2)
    padded[ 2 : rows + 1, 2 : columns + 1 ] = binary;

    # Remove interior pixels with all 4-connected neighbors.
    N = circshift( padded, [  0  1 ] );
    S = circshift( padded, [  0 -1 ] );
    E = circshift( padded, [ -1  0 ] );
    W = circshift( padded, [  1  0 ] );
    boundary_image = padded .- (( padded .+ N .+ S .+ E .+ W) .== 5 );

    # To prevent reallocating boundary, we need to initialize it.
    boundary_size = sum( boundary_image);
    boundary = sizehint!(Vector(), boundary_size)

    # Scan for the first pixel, Left-to-Right & Top-to-Bottom.
    initial_entry = (0,0)
    for  j = 1:columns, i = 1:rows
        if binary[ i, j ] == 1
            initial_entry = (j,i) .+ 1
           break;
       end
    end
    # Set this pixel ( w/ padded offset ) as the initial entry point.

    # Designate a directional offset array for search positions.
    # [ 2 ][ 3 ][ 4 ]
    # [ 1 ][ X ][ 5 ]
    # [ 8 ][ 7 ][ 6 ]
    # Column 1: x-axis offset // Column 2: y-axis offset
    neighborhood = [ -1 0; -1 -1; 0 -1; 1 -1; 1 0; 1 1; 0 1; -1 1 ];
    exit_direction = [ 7 7 1 1 3 3 5 5 ];

    # Find the first point in the boundary, Moore-Neighbor of entry point.
    n = 0
    initial_position = -1
    for n = 1:8 # 8-connected neighborhood
        c = initial_entry .+ neighborhood[n,:];
        if padded[ c[ 2 ], c[ 1 ] ]
            initial_position = c;
            break;
        end
    end

    # Set next direction based on found pixel ( i.e. 3 -> 1).
    initial_direction = exit_direction[n];

    # Start the boundary set with this pixel.
    push!(boundary,  initial_position);

    # Initialize variables for boundary search.
    position = initial_position;
    direction = initial_direction;
    # return a list of the ordered boundary pixels.
    while true

        # Find the next neighbor with a clockwise search.
        for n in circshift( collect(1:8), [1-direction ] )
            c = position .+ neighborhood[ n, : ];
            if padded[ c[ 2 ], c[1 ] ]
                position = c;
                break;
            end
        end

        # Neighbor found, save its information.
        direction = exit_direction[n];
        push!(boundary, position)

        # Entered the initial pixel the same way twice, the end.
        if  (position == initial_position)  && ( direction == initial_direction )
           break;
        end
    end
    return [[j[2]-1;j[1]-1] for j in boundary]
end
