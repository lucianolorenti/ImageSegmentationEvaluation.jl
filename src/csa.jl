using OffsetArrays
using LightGraphs
using StaticGraphs
mutable struct IterationStats
    double_pushes::Integer
    relabelings::Integer
    pushes::Integer
    refinements::Integer 
end
function clear!(it::IterationStats)
    it.double_pushes=0
    it.relabelings=0
    it.pushes=0
    it.refinements=0
end
function IterationStats()
    return IterationStats(0,0,0,0)
end
mutable struct CSA
    g::StaticDiGraph
    num_left_nodes::Int64
    largest_scaled_cost_magnitude::Int64
    cost_scaling_factor::Integer
    matched_arc::Vector{Int64}
    price::OffsetVector{Int64}
    matched_node::OffsetVector{Int64}
    epsilon::Int64
    slack_relabeling_price::Integer
    active_nodes::Vector{Int64}
    alpha::Int64
    price_lower_bound::Int64
    total_excess::Int64
    scaled_arc_cost::Vector{Int64}
    iteration_stats::IterationStats
    iteration_stats_list::Vector{IterationStats}
    success::Bool
    kMinEpsilon::Integer
end
const kNilArc = typemax(Int64)
const kNilNode = typemax(Int64)
function CSA(g::StaticDiGraph, w::Vector{Float64}, num_left_nodes::Integer, kMinEpsilon::Integer)
    local scaling_factor = num_left_nodes + 1
    local scaled_arc_cost = round.(Int64,w*scaling_factor)
    return CSA(g,
               num_left_nodes,
               maximum(abs.(scaled_arc_cost)),
               scaling_factor,
               zeros(Int64,num_left_nodes),
               OffsetVector(Int64,num_left_nodes+1:(2 * num_left_nodes )),
               OffsetVector(Int64,num_left_nodes+1:(2 * num_left_nodes )),
               0,
               0,
               Vector{Int64}(),
               scaling_factor,
               0,
               0,
               scaled_arc_cost,
               IterationStats(),
               Vector{IterationStats}(),
               false,
               kMinEpsilon)
end


"""
Computes the value of the bound on price reduction for an iteration, given the old and new values of epsilon_.  Because the
expression computed here is used in at least one place where we want an additional factor in the denominator, we take that factor as an argument. If extra_divisor == 1, this function computes of   the function B() discussed above.
Avoids overflow in computing the bound, and sets *in_range = false if the value of the bound doesn't fit in CostValue.
"""
function  price_change_bound(csa::CSA, old_epsilon::Int64, new_epsilon::Int64)
    local n = nv(csa.g)
    #=
    // We work in double-precision floating point to determine whether
    // we'll overflow the integral CostValue type's range of
    // representation. Switching between integer and double is a
    // rather expensive operation, but we do this only twice per
    // scaling iteration, so we can afford it rather than resort to
    // complex and subtle tricks within the bounds of integer
    // arithmetic.
    //
    // You will want to read the comments above about
    // price_lower_bound_ and slack_relabeling_price_, and have a
    // pencil handy. :-)=#
    const result = max(1, n / 2 - 1) *    ( old_epsilon + new_epsilon);
    const limit  = typemax(Int64)
    if (result > limit) 
##      // Our integer computations could overflow.
        return (false, typemax(Int64))
     else 
        # Don't touch *in_range; other computations could already have set it to false and we don't want to overwrite that result.
        return (true, round(Integer,result))
    end
end
function new_epsilon(csa::CSA, current_epsilon)
    return max(current_epsilon / csa.alpha, csa.kMinEpsilon);
end

function update_epsilon!(csa::CSA)
    local new_epsilon_ = new_epsilon(csa, csa.epsilon);
    (_,csa.slack_relabeling_price) = price_change_bound(csa, round(Int64,csa.epsilon), round(Int64,new_epsilon_))
    csa.epsilon =  round(Int64,new_epsilon_);
    #VLOG(3) << "Updated: epsilon_ == " << epsilon_;
    #VLOG(4) << "slack_relabeling_price_ == " << slack_relabeling_price_;
    @assert csa.slack_relabeling_price >= 0
    return true
end
function finalize_setup!(csa::CSA)
    local incidence_precondition_satisfied = true;
    #= epsilon_ must be greater than kMinEpsilon so that in the case
    where the largest arc cost is zero, we still do a Refine()
    iteration.=#
    csa.epsilon = max(csa.largest_scaled_cost_magnitude, csa.kMinEpsilon + 1);
    # Initialize left-side node-indexed arrays and check incidence precondition.
    for node=1:csa.num_left_nodes
        csa.matched_arc[node] = kNilArc;
        if (outdegree(csa.g, node) == 0)
            incidence_precondition_satisfied = false;
        end
    end
    #Initialize right-side node-indexed arrays. Example: prices are stored only for right-side nodes.
    for node=csa.num_left_nodes+1:nv(csa.g)        
        csa.price[node] = 0;
        csa.matched_node[node] = kNilNode;
    end
        
    local in_range = true;
    local double_price_lower_bound = 0.0;
    local old_error_parameter = csa.epsilon;
    local new_error_parameter = 0    
    while (new_error_parameter != csa.kMinEpsilon)
        new_error_parameter = new_epsilon(csa, old_error_parameter);
        (in_range, price_bound) =  (price_change_bound( csa, round(Int64,old_error_parameter), round(Int64,new_error_parameter)))
        double_price_lower_bound -=  2.0 * price_bound
        old_error_parameter = new_error_parameter;
    end
    const  limit = -convert(Float64, typemax(Int64))
    if (double_price_lower_bound < limit) 
        in_range = false;
        csa.price_lower_bound = -typemax(Int64)
    else 
        csa.price_lower_bound = convert(Integer,double_price_lower_bound);
    end
    if (!in_range) 
        warn("""Price change bound exceeds range of representable 
                 costs; arithmetic overflow is not ruled out and 
                 infeasibility might go undetected.""");
    end
    return  incidence_precondition_satisfied
end
"""
    Returns the arc through which the given node is matched.
    """
function get_assignment_arc(csa::CSA, left_node::Int64)
    @assert left_node<=csa.num_left_nodes
    return csa.matched_arc[left_node]
end
"""
```
function get_mate(csa::CSA, left_node::Int64)
```
Returns the node to which the given node is matched.
"""
function get_mate(csa::CSA, left_node::Int64)
    @assert left_node<=csa.num_left_nodes
    local matching_arc = get_assignment_arc(csa,left_node);
    @assert kNilArc!= matching_arc
    return csa.g.f_vec[matching_arc]
end
"""
```
function saturate_negative_arcs!(csa::CSA)
```
Saturates all negative-reduced-cost arcs at the beginning of each scaling iteration. Note that according to the asymmetric
definition of admissibility, this action is different from  saturating all admissible arcs (which we never do). All negative
arcs are admissible, but not all admissible arcs are negative. It  is alwsys enough to saturate only the negative ones.

// There exists a price function such that the admissible arcs at the
// beginning of an iteration are exactly the reverse arcs of all
// matching arcs. Saturating all admissible arcs with respect to that
// price function therefore means simply unmatching every matched
// node.
//
// In the future we will price out arcs, which will reduce the set of
// nodes we unmatch here. If a matching arc is priced out, we will not
// unmatch its endpoints since that element of the matching is
// guaranteed not to change.
"""
function saturate_negative_arcs!(csa::CSA)
    csa.total_excess = 0;
    #Iterate over left_nodes
    for node=1:csa.num_left_nodes
        if (isactive(csa,node)) 
            # This can happen in the first iteration when nothing is matched yet.
            csa.total_excess += 1;
        else 
            # We're about to create a unit of excess by unmatching these nodes.
            csa.total_excess += 1;
            local mate = get_mate(csa, node);
            csa.matched_arc[node] =  kNilArc;
            csa.matched_node[mate] = kNilNode;
        end
    end
end
function isactive(csa::CSA, left_node::Int64)
  @assert left_node<=csa.num_left_nodes
  return csa.matched_arc[left_node] == kNilArc;
end
function initialize_active_node_container!(csa::CSA) 
    @assert isempty(csa.active_nodes)
    for node=1:csa.num_left_nodes        
        if isactive(csa,node)
            push!(csa.active_nodes,node);
        end
    end
end
"""
Returns the partial reduced cost of the given arc.
"""
function partial_reduced_cost(csa::CSA, head::Integer, arc::Int64)
    return csa.scaled_arc_cost[arc] - csa.price[head];
end

"""
// Computes best_arc, the minimum reduced-cost arc incident to
// left_node and admissibility_gap, the amount by which the reduced
// cost of best_arc must be increased to make it equal in reduced cost to another residual arc incident to left_node.
//
// Precondition: left_node is unmatched and has at least one incident
// arc. This allows us to simplify the code. The debug-only
// counterpart to this routine is LinearSumAssignment::ImplicitPrice()
// and it assumes there is an incident arc but does not assume the
// node is unmatched. The condition that each left node has at least
// one incident arc is explicitly computed during FinalizeSetup().
//
// This function is large enough that our suggestion that the compiler
// inline it might be pointless.
"""
function best_arc_and_gap(csa::CSA, left_node::Int64)
    @assert isactive(csa,left_node)
    @assert csa.epsilon > 0
    local neigh = outneighbors(csa.g, left_node)
    local best_arc = neigh.offset + 1
    local min_partial_reduced_cost = partial_reduced_cost(csa, neigh[1], best_arc);
    #We choose second_min_partial_reduced_cost so that in the case of the largest possible gap (which results from a left-side node
    #ith only a single incident residual arc), the corresponding  right-side node will be relabeled by an amount that exactly
    # matches slack_relabeling_price_.
    const  max_gap = csa.slack_relabeling_price - csa.epsilon;
    local  second_min_partial_reduced_cost =  min_partial_reduced_cost + max_gap;
    for j=2:length(neigh)
        local arc = neigh.offset + j
        local partial_reduced_cost_ = partial_reduced_cost(csa, neigh[j], arc);
        if (partial_reduced_cost_ < second_min_partial_reduced_cost) 
            if (partial_reduced_cost_ < min_partial_reduced_cost) 
                best_arc = arc;
                second_min_partial_reduced_cost = min_partial_reduced_cost;
                min_partial_reduced_cost = partial_reduced_cost_;
            else 
                second_min_partial_reduced_cost = partial_reduced_cost_;
            end
        end
    end
    local gap = min(second_min_partial_reduced_cost - min_partial_reduced_cost, max_gap);
    @assert gap>=0
    return (best_arc, gap)
end


"""
Returns true for success, false for infeasible.
"""
function double_push!(csa::CSA, source::Int64)
    @assert csa.num_left_nodes>=source
    @assert isactive(csa,source)
  (best_arc, gap) = best_arc_and_gap(csa, source);
    # Now we have the best arc incident to source, i.e., the one with minimum reduced cost. Match that arc, unmatching its head if
    # necessary.
    if (best_arc == kNilArc) 
      return false;
  end
    local  new_mate = csa.g.f_vec[best_arc]
    local to_unmatch = csa.matched_node[new_mate];
    if (to_unmatch != kNilNode) 
      #Unmatch new_mate from its current mate, pushing the unit of flow back to a node on the left side as a unit of excess.
      csa.matched_arc[to_unmatch] = kNilArc;
      push!(csa.active_nodes, to_unmatch)
      # This counts as a double push.
      csa.iteration_stats.double_pushes += 1;
  else
      # We are about to increase the cardinality of the matching.
      csa.total_excess -= 1;
      # This counts as a single push.
      csa.iteration_stats.pushes += 1;
    end
    
    csa.matched_arc[source] = best_arc;
    csa.matched_node[new_mate] = source;
    #Finally, relabel new_mate.
    csa.iteration_stats.relabelings += 1;
    local new_price = csa.price[new_mate] - gap - csa.epsilon;
    csa.price[new_mate] = new_price;
    return new_price >= csa.price_lower_bound;
end

function refine!(csa::CSA)
  saturate_negative_arcs!(csa);
  initialize_active_node_container!(csa);
    while (csa.total_excess > 0)
      # Get an active node (i.e., one with excess == 1) and discharge it using DoublePush.
        # Using a queue
        local node = pop!(csa.active_nodes)
      if (!double_push!(csa, node))
          #= Infeasibility detected.
          //
          // If infeasibility is detected after the first iteration, we
          // have a bug. We don't crash production code in this case but
          // we know we're returning a wrong answer so we we leave a
          // message in the logs to increase our hope of chasing down the
          // problem.=#
          warn("Infeasibility detection triggered after first iteration found  a feasible assignment!");
      return false;
    end
  end
  @assert isempty(csa.active_nodes)
  csa.iteration_stats.refinements += 1;
  return true;
end

"""
Returns the original arc cost for use by a client that's iterating over the optimum assignment.
"""
function arc_cost(csa::CSA, arc::Int64)
    @assert 0 == csa.scaled_arc_cost[arc] % csa.cost_scaling_factor;
    return csa.scaled_arc_cost[arc] / csa.cost_scaling_factor;
end
"""
Returns the arc through which the given node is matched.
"""
function assignment_arc(csa::CSA, left_node::Int64)
    @assert left_node <= csa.num_left_nodes
    return csa.matched_arc[left_node];
end
"""

Returns the cost of the assignment arc incident to the given node.
"""
function assignment_cost(csa::CSA, node::Int64)
    return arc_cost(csa,assignment_arc(csa,node));
end
function cost(csa::CSA)
    #it is illegal to call this method unless we successfully computed an optimum assignment.
    @assert csa.success
    local cost_ = 0
    for node=1:csa.num_left_nodes
        cost_ += assignment_cost(csa, node);
    end
  return cost_;
end
function main(csa::CSA)
    local ok = true
    local incidence_precondition_satisfied = finalize_setup!(csa)
    ok = ok && incidence_precondition_satisfied;
    #DCHECK(!ok || EpsilonOptimal());
    while (ok && (csa.epsilon > csa.kMinEpsilon))
        
        @time ok = ok && update_epsilon!(csa);
        @time ok = ok && refine!(csa);
        push!(csa.iteration_stats_list, csa.iteration_stats)
#    DCHECK(!ok || EpsilonOptimal());
 #   DCHECK(!ok || AllMatched());
    end
    csa.success=ok
end

function csa(g::StaticDiGraph, d::Vector{T}, left_nodes::Integer, kminEpsilon::Integer=1) where T
    local c = CSA(g, d, left_nodes, kminEpsilon)
    main(c)
    return c
end
#=
using Base.Test
@testset "4x4 Matrix" begin

  const  kNumSources = 4;
  const  kNumTargets = 4;
     kCost = [90 76 75 80;
                    35 85 55 65;
                    125 95 90 105;
                    45 110 95 115]
     kExpectedCost = kCost[1,4] + kCost[2,3] + kCost[3,2] + kCost[4,1];
     g = DiGraph(kNumSources + kNumTargets)
     w = Dict{Edge,Float64}()
    for source=1:kNumSources
        for target=1:kNumTargets
            add_edge!(g,source, kNumSources+target)
            w[Edge(source, kNumSources+target)]= kCost[source,target]
        end
    end
    local assignment = csa(g,w, kNumSources)
    local total_cost = cost(assignment)
    @test assignment.success == true
    @test kExpectedCost==total_cost
end

=#
