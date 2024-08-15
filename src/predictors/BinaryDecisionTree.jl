# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    BinaryDecisionTree{K,V}(
        feat_id::Int,
        feat_value::K,
        lhs::Union{V,BinaryDecisionTree{K,V}},
        rhs::Union{V,BinaryDecisionTree{K,V}},
    )

An [`AbstractPredictor`](@ref) that represents a binary decision tree.

 * If `x[feat_id] <= feat_value`, then return `lhs`
 * If `x[feat_id] > feat_value`, then return `rhs`

## Example

To represent the tree `x[1] <= 0.0 ? -1 : (x[1] <= 1.0 ? 0 : 1)`, do:

```jldoctest doc_decision_tree
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:1]);

julia> f = MathOptAI.BinaryDecisionTree{Float64,Int}(
           1,
           0.0,
           -1,
           MathOptAI.BinaryDecisionTree{Float64,Int}(1, 1.0, 0, 1),
       )
BinaryDecisionTree{Float64,Int64} [leaves=3, depth=2]

julia> y = MathOptAI.add_predictor(model, f, x)
1-element Vector{VariableRef}:
 moai_BinaryDecisionTree_value

julia> print(model)
Feasibility
Subject to
 moai_BinaryDecisionTree_z[1] + moai_BinaryDecisionTree_z[2] + moai_BinaryDecisionTree_z[3] = 1
 moai_BinaryDecisionTree_z[1] - moai_BinaryDecisionTree_z[3] + moai_BinaryDecisionTree_value = 0
 moai_BinaryDecisionTree_z[1] --> {x[1] ≤ 0}
 moai_BinaryDecisionTree_z[2] --> {x[1] ≤ 1}
 moai_BinaryDecisionTree_z[1] binary
 moai_BinaryDecisionTree_z[2] binary
 moai_BinaryDecisionTree_z[3] binary
 moai_BinaryDecisionTree_z[2] --> {x[1] ≥ 0}
 moai_BinaryDecisionTree_z[3] --> {x[1] ≥ 0}
 moai_BinaryDecisionTree_z[3] --> {x[1] ≥ 1}
```
"""
struct BinaryDecisionTree{K,V}
    feat_id::Int
    feat_value::K
    lhs::Union{V,BinaryDecisionTree{K,V}}
    rhs::Union{V,BinaryDecisionTree{K,V}}
end

function Base.show(io::IO, predictor::BinaryDecisionTree{K,V}) where {K,V}
    paths = _tree_to_paths(predictor)
    leaves, depth = length(paths), maximum(length.(paths))
    return print(io, "BinaryDecisionTree{$K,$V} [leaves=$leaves, depth=$depth]")
end

function add_predictor(
    model::JuMP.Model,
    predictor::BinaryDecisionTree,
    x::Vector;
    atol::Float64 = 0.0,
)
    paths = _tree_to_paths(predictor)
    z = JuMP.@variable(
        model,
        [1:length(paths)],
        binary = true,
        base_name = "moai_BinaryDecisionTree_z",
    )
    JuMP.@constraint(model, sum(z) == 1)
    y = JuMP.@variable(model, base_name = "moai_BinaryDecisionTree_value")
    y_expr = JuMP.AffExpr(0.0)
    for (zi, (leaf, path)) in zip(z, paths)
        JuMP.add_to_expression!(y_expr, leaf, zi)
        for (id, value, branch) in path
            if branch
                JuMP.@constraint(model, zi --> {x[id] <= value})
            else
                JuMP.@constraint(model, zi --> {x[id] >= value + atol})
            end
        end
    end
    JuMP.@constraint(model, y == y_expr)
    return [y]
end

function _tree_to_paths(predictor::BinaryDecisionTree{K,V}) where {K,V}
    paths = Pair{V,Vector{Tuple{Int,K,Bool}}}[]
    _tree_to_paths(paths, Tuple{Int,K,Bool}[], predictor)
    return paths
end

function _tree_to_paths(
    paths::Vector{Pair{V,Vector{Tuple{Int,K,Bool}}}},
    current_path::Vector{Tuple{Int,K,Bool}},
    node::BinaryDecisionTree{K,V},
) where {K,V}
    left_path = vcat(current_path, (node.feat_id, node.feat_value, true))
    _tree_to_paths(paths, left_path, node.lhs)
    right_path = vcat(current_path, (node.feat_id, node.feat_value, false))
    _tree_to_paths(paths, right_path, node.rhs)
    return
end

function _tree_to_paths(
    paths::Vector{Pair{V,Vector{Tuple{Int,K,Bool}}}},
    current_path::Vector{Tuple{Int,K,Bool}},
    node::V,
) where {K,V}
    push!(paths, node => current_path)
    return
end
