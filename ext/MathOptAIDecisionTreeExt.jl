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
        model::JuMP.Model,
        predictor::DecisionTree.Root,
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

julia> y = MathOptAI.add_predictor(model, ml_model, x)
1-element Vector{VariableRef}:
 moai_BinaryDecisionTree_value
```
"""
function MathOptAI.add_predictor(
    model::JuMP.Model,
    predictor::DecisionTree.Root,
    x::Vector;
    kwargs...,
)
    inner_predictor = _tree_or_leaf(predictor.node)
    return MathOptAI.add_predictor(model, inner_predictor, x; kwargs...)
end

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
