# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    BinaryDecisionTree{K,V}(
        feat_id::Int,
        feat_value::K,
        lhs::Union{V,BinaryDecisionTree{K,V}},
        rhs::Union{V,BinaryDecisionTree{K,V}},
        atol::Float64 = 1e-6,
    )

An [`AbstractPredictor`](@ref) that represents a binary decision tree.

 * If `x[feat_id] <= feat_value - atol`, then return `lhs`
 * If `x[feat_id] >= feat_value`, then return `rhs`

## Example

To represent the tree `x[1] <= 0.0 ? -1 : (x[1] <= 1.0 ? 0 : 1)`, do:

```jldoctest
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

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
1-element Vector{VariableRef}:
 moai_BinaryDecisionTree_value[1]

julia> formulation
BinaryDecisionTree{Float64,Int64} [leaves=3, depth=2]
├ variables [4]
│ ├ moai_BinaryDecisionTree_value[1]
│ ├ moai_BinaryDecisionTree_z[1]
│ ├ moai_BinaryDecisionTree_z[2]
│ └ moai_BinaryDecisionTree_z[3]
└ constraints [7]
  ├ moai_BinaryDecisionTree_z[1] + moai_BinaryDecisionTree_z[2] + moai_BinaryDecisionTree_z[3] = 1
  ├ moai_BinaryDecisionTree_z[1] --> {x[1] ≤ -1.0e-6}
  ├ moai_BinaryDecisionTree_z[2] --> {x[1] ≥ 0}
  ├ moai_BinaryDecisionTree_z[2] --> {x[1] ≤ 0.999999}
  ├ moai_BinaryDecisionTree_z[3] --> {x[1] ≥ 0}
  ├ moai_BinaryDecisionTree_z[3] --> {x[1] ≥ 1}
  └ moai_BinaryDecisionTree_z[1] - moai_BinaryDecisionTree_z[3] + moai_BinaryDecisionTree_value[1] = 0
```
"""
struct BinaryDecisionTree{K,V} <: AbstractPredictor
    feat_id::Int
    feat_value::K
    lhs::Union{V,BinaryDecisionTree{K,V}}
    rhs::Union{V,BinaryDecisionTree{K,V}}
    atol::Float64

    function BinaryDecisionTree{K,V}(
        feat_id::Int,
        feat_value::K,
        lhs::Union{V,BinaryDecisionTree{K,V}},
        rhs::Union{V,BinaryDecisionTree{K,V}},
        atol::Float64 = 1e-6,
    ) where {K,V}
        return new{K,V}(feat_id, feat_value, lhs, rhs, atol)
    end

    function BinaryDecisionTree(
        feat_id::Int,
        feat_value::K,
        lhs::Union{V,BinaryDecisionTree{K,V}},
        rhs::Union{V,BinaryDecisionTree{K,V}},
        atol::Float64 = 1e-6,
    ) where {K,V}
        return new{K,V}(feat_id, feat_value, lhs, rhs, atol)
    end
end

function Base.show(io::IO, predictor::BinaryDecisionTree{K,V}) where {K,V}
    paths = _tree_to_paths(predictor)
    leaves, depth = length(paths), maximum(length.(paths))
    return print(io, "BinaryDecisionTree{$K,$V} [leaves=$leaves, depth=$depth]")
end

output_size(::BinaryDecisionTree, input_size) = (1,)

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::BinaryDecisionTree,
    x::Vector,
)
    paths = _tree_to_paths(predictor)
    z = add_variables(model, x, length(paths), "moai_BinaryDecisionTree_z")
    JuMP.set_binary.(z)
    c = JuMP.@constraint(model, sum(z) == 1)
    y = only(add_variables(model, x, 1, "moai_BinaryDecisionTree_value"))
    y_expr = JuMP.AffExpr(0.0)
    formulation = Formulation(predictor, Any[y; z], Any[c])
    for (zi, (leaf, path)) in zip(z, paths)
        JuMP.add_to_expression!(y_expr, leaf, zi)
        for (id, value, branch, atol) in path
            c = if branch
                JuMP.@constraint(model, zi --> {x[id] <= value - atol})
            else
                JuMP.@constraint(model, zi --> {x[id] >= value})
            end
            push!(formulation.constraints, c)
        end
    end
    push!(formulation.constraints, JuMP.@constraint(model, y == y_expr))
    return [y], formulation
end

function _tree_to_paths(predictor::BinaryDecisionTree{K,V}) where {K,V}
    paths = Pair{V,Vector{Tuple{Int,K,Bool,Float64}}}[]
    _tree_to_paths(paths, Tuple{Int,K,Bool,Float64}[], predictor)
    return paths
end

function _tree_to_paths(
    paths::Vector{Pair{V,Vector{Tuple{Int,K,Bool,Float64}}}},
    current_path::Vector{Tuple{Int,K,Bool,Float64}},
    node::BinaryDecisionTree{K,V},
) where {K,V}
    left_path =
        vcat(current_path, (node.feat_id, node.feat_value, true, node.atol))
    _tree_to_paths(paths, left_path, node.lhs)
    right_path = vcat(current_path, (node.feat_id, node.feat_value, false, 0.0))
    _tree_to_paths(paths, right_path, node.rhs)
    return
end

function _tree_to_paths(
    paths::Vector{Pair{V,Vector{Tuple{Int,K,Bool,Float64}}}},
    current_path::Vector{Tuple{Int,K,Bool,Float64}},
    node::V,
) where {K,V}
    push!(paths, node => current_path)
    return
end
