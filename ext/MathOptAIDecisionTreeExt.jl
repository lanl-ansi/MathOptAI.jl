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
        x::Vector;
        config::Dict = Dict{Any,Any}(),
    )

Add a binary decision tree from DecisionTree.jl to `model`.

## Example
"""
function MathOptAI.add_predictor(
    model::JuMP.Model,
    predictor::DecisionTree.Root{K,V},
    x::Vector;
    config::Dict = Dict{Any,Any}(),
) where {K,V}
    inner_predictor = MathOptAI.BinaryDecisionTree(predictor.node)
    return MathOptAI.add_predictor(model, inner_predictor, x)
end

function MathOptAI.BinaryDecisionTree(
    node::DecisionTree.Node{K,V},
) where {K,V}
    return MathOptAI.BinaryDecisionTree{K,V}(
        node.featid,
        node.featval,
        _tree_or_leaf(node.left),
        _tree_or_leaf(node.right),
    )
end

_tree_or_leaf(node::DecisionTree.Node) = MathOptAI.BinaryDecisionTree(node)
_tree_or_leaf(node::DecisionTree.Leaf) = node.majority

end  # module
