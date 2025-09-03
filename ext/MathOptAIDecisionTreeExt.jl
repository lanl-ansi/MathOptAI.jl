# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAIDecisionTreeExt

import DecisionTree
import JuMP
import MathOptAI

"""
    MathOptAI.build_predictor(
        predictor::Union{
            DecisionTree.Ensemble,
            DecisionTree.DecisionTreeClassifier,
            DecisionTree.Leaf,
            DecisionTree.Node,
            DecisionTree.Root,
        },
    )

Convert a binary decision tree from DecisionTree.jl to a
[`BinaryDecisionTree`](@ref).

## Example

```jldoctest
julia> using JuMP, MathOptAI, DecisionTree

julia> truth(x::Vector) = x[1] <= 0.5 ? -2 : (x[2] <= 0.3 ? 3 : 4)
truth (generic function with 1 method)

julia> features = abs.(sin.((1:10) .* (3:4)'));

julia> size(features)
(10, 2)

julia> labels = truth.(Vector.(eachrow(features)));

julia> tree = DecisionTree.build_tree(labels, features)
Decision Tree
Leaves: 3
Depth:  2

julia> model = Model();

julia> @variable(model, 0 <= x[1:2] <= 1);

julia> y, _ = MathOptAI.add_predictor(model, tree, x);

julia> y
1-element Vector{VariableRef}:
 moai_BinaryDecisionTree_value

julia> MathOptAI.build_predictor(tree)
BinaryDecisionTree{Float64,Int64} [leaves=3, depth=2]
```
"""
function MathOptAI.build_predictor(
    predictor::Union{
        DecisionTree.Ensemble,
        DecisionTree.DecisionTreeClassifier,
        DecisionTree.Leaf,
        DecisionTree.Node,
        DecisionTree.Root,
    };
    kwargs...,
)
    return _build_predictor(predictor; kwargs...,)
end

function _build_predictor(p::DecisionTree.Root; kwargs...)
    return _build_predictor(p.node; kwargs...)
end

function _build_predictor(p::DecisionTree.DecisionTreeClassifier; kwargs...)
    return _build_predictor(p.root; kwargs...)
end

function _build_predictor(
    node::DecisionTree.Node{K,V};
    atol::Float64 = 1e-6,
) where {K,V}
    return MathOptAI.BinaryDecisionTree{K,V}(
        node.featid,
        node.featval,
        _build_predictor(node.left; atol),
        _build_predictor(node.right; atol),
        atol,
    )
end

_build_predictor(node::DecisionTree.Leaf; kwargs...) = node.majority

function _build_predictor(
    node::DecisionTree.Ensemble{K,V};
    kwargs...,
) where {K,V}
    trees = _build_predictor.(node.trees; kwargs...)
    weights = fill(1 / length(trees), length(trees))
    return MathOptAI.AffineCombination(trees, weights, [0.0])
end

end  # module MathOptAIDecisionTreeExt
