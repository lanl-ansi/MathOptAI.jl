# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAIDecisionTreeExt

import DecisionTree
import JuMP
import MathOptAI

"""
    MathOptAI.add_predictor(
        model::JuMP.AbstractModel,
        predictor::Union{DecisionTree.Root,DecisionTree.DecisionTreeClassifier},
        x::Vector,
    )

Add a binary decision tree from DecisionTree.jl to `model`.

## Example

```jldoctest
julia> using JuMP, MathOptAI, DecisionTree

julia> truth(x::Vector) = x[1] <= 0.5 ? -2 : (x[2] <= 0.3 ? 3 : 4)
truth (generic function with 1 method)

julia> features = abs.(sin.((1:10) .* (3:4)'));

julia> size(features)
(10, 2)

julia> labels = truth.(Vector.(eachrow(features)));

julia> ml_model = DecisionTree.build_tree(labels, features)
Decision Tree
Leaves: 3
Depth:  2

julia> model = Model();

julia> @variable(model, 0 <= x[1:2] <= 1);

julia> y, _ = MathOptAI.add_predictor(model, ml_model, x);

julia> y
1-element Vector{VariableRef}:
 moai_BinaryDecisionTree_value
```
"""
function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
    predictor::Union{DecisionTree.Root,DecisionTree.DecisionTreeClassifier},
    x::Vector,
)
    inner_predictor = MathOptAI.build_predictor(predictor)
    return MathOptAI.add_predictor(model, inner_predictor, x)
end

"""
    MathOptAI.build_predictor(predictor::DecisionTree.Root)

Convert a binary decision tree from DecisionTree.jl to a
[`BinaryDecisionTree`](@ref).

## Example

```jldoctest
julia> using MathOptAI, DecisionTree

julia> truth(x::Vector) = x[1] <= 0.5 ? -2 : (x[2] <= 0.3 ? 3 : 4)
truth (generic function with 1 method)

julia> features = abs.(sin.((1:10) .* (3:4)'));

julia> size(features)
(10, 2)

julia> labels = truth.(Vector.(eachrow(features)));

julia> ml_model = DecisionTree.build_tree(labels, features)
Decision Tree
Leaves: 3
Depth:  2

julia> MathOptAI.build_predictor(ml_model)
BinaryDecisionTree{Float64,Int64} [leaves=3, depth=2]
```
"""
MathOptAI.build_predictor(p::DecisionTree.Root) = _tree_or_leaf(p.node)

function MathOptAI.build_predictor(p::DecisionTree.DecisionTreeClassifier)
    return MathOptAI.build_predictor(p.root)
end

MathOptAI.build_predictor(p::DecisionTree.Node) = _tree_or_leaf(p)

function _tree_or_leaf(node::DecisionTree.Node{K,V}) where {K,V}
    return MathOptAI.BinaryDecisionTree{K,V}(
        node.featid,
        node.featval,
        _tree_or_leaf(node.left),
        _tree_or_leaf(node.right),
    )
end

_tree_or_leaf(node::DecisionTree.Leaf) = node.majority

end  # module
