using ImageSegmentationEvaluation
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end
@testset "Boundary extraction" begin
    img = [0 0 0 0 0 0 0 0 0;
           0 0 0 0 0 0 0 0 0;
           0 0 1 1 1 1 1 0 0;
           0 0 1 1 1 1 1 0 0;
           0 0 1 1 1 1 1 0 0;
           0 0 1 1 1 1 1 0 0;
           0 0 1 1 1 1 1 0 0;
           0 0 1 1 1 1 1 0 0;
           0 0 0 0 0 0 0 0 0;
           0 0 0 0 0 0 0 0 0]
    
    expected = [0 0 0 0 0 0 0 0 0;
                0 0 1 1 1 1 1 0 0;
                0 1 1 1 1 1 1 1 0;
                0 1 1 0 0 0 1 1 0;
                0 1 1 0 0 0 1 1 0;
                0 1 1 0 0 0 1 1 0;
                0 1 1 0 0 0 1 1 0;
                0 1 1 1 1 1 1 1 0;
                0 0 1 1 1 1 1 0 0;
                0 0 0 0 0 0 0 0 0]
    @test expected == boundary_map(BoundaryGradient(), img)
    
    expected = [ 0  0  0  0  0  0  0  0  0;
                 0  1  1  1  1  1  1  0  0;
                 0  1  0  0  0  0  1  0  0;
                 0  1  0  0  0  0  1  0  0;
                 0  1  0  0  0  0  1  0  0;
                 0  1  0  0  0  0  1  0  0;
                 0  1  0  0  0  0  1  0  0;
                 0  1  1  1  1  1  1  0  0;
                 0  0  0  0  0  0  0  0  0;
                 0  0  0  0  0  0  0  0  0]
    @test expected == boundary_map(BoundaryShift(), img)
    display(boundary_map(BoundaryMooreTracing(), img)  )
    @test expected == boundary_map(BoundaryMooreTracing(), img) 
    
    
end
@testset "Boundary Displacement Error" begin
    function distances(B1, B2)
        d=[]
        for j = 1:size(B1,1)
            p = B1[j,:]
            push!(d, minimum([norm(B2[k,:]-p) for k = 1:size(B2,1)]))
        end
        return d
    end
    img = [0 0 0 0 0;
           0 1 1 1 0;
           0 1 1 1 0;
           0 1 1 1 0;
           0 0 0 0 0]
    gt = [0 0 0 0 0;
          0 0 1 1 0;
          0 0 1 1 0;
          0 0 1 1 0;
          0 0 0 0 0]

    img_boundary =  [0 1 1 1 0;
                     1 1 1 1 1;
                     1 1 0 1 1;
                     1 1 1 1 1;
                     0 1 1 1 0]
    gt_boundary = []
    display(Integer.(boundary_map(BoundaryGradient(), gt)))
    B1 = ind2sub(size(img),find(img.==1))
    B1 = hcat(B1...)
    B2 = ind2sub(size(gt), find(gt.==1))
    B2 = hcat(B2...)
    
    d1 = mean(distances(B1, B2))
    d2 = mean(distances(B2, B1))
    println(d1)
    println(d2)
    @test evaluate(BoundaryDisplacementError(), img, gt) == (d1+d2)/2
end

# write your own tests here
@test 1 == 2
