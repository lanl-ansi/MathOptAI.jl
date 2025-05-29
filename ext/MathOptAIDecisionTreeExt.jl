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
    MathOptAI.add_predictor(
        model::JuMP.AbstractModel,
        predictor::Union{
            DecisionTree.Root,
            DecisionTree.DecisionTreeClassifier,
            DecisionTree.Ensemble,
        },
        x::Vector,
    )

Add a binary decision tree (or random forest) from DecisionTree.jl to `model`.

## Example

```jldoctest
julia> using JuMP, MathOptAI, DecisionTree

julia> truth(x::Vector) = x[1] <= 0.5 ? -2 : (x[2] <= 0.3 ? 3 : 4)
truth (generic function with 1 method)

julia> features = abs.(sin.((1:10) .* (3:4)'));

julia> size(features)
(10, 2)

julia> labels = truth.(Vector.(eachrow(features)));

julia> predictor = DecisionTree.build_tree(labels, features)
Decision Tree
Leaves: 3
Depth:  2

julia> model = Model();

julia> @variable(model, 0 <= x[1:2] <= 1);

julia> y, _ = MathOptAI.add_predictor(model, predictor, x);

julia> y
1-element Vector{VariableRef}:
 moai_BinaryDecisionTree_value
```
"""
function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
    predictor::Union{
        DecisionTree.Root,
        DecisionTree.DecisionTreeClassifier,
        DecisionTree.Ensemble,
    },
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

julia> tree = DecisionTree.build_tree(labels, features)
Decision Tree
Leaves: 3
Depth:  2

julia> predictor = MathOptAI.build_predictor(tree)
BinaryDecisionTree{Float64,Int64} [leaves=3, depth=2]
```
"""
function MathOptAI.build_predictor(p::DecisionTree.Root)
    return MathOptAI.build_predictor(p.node)
end

function MathOptAI.build_predictor(p::DecisionTree.DecisionTreeClassifier)
    return MathOptAI.build_predictor(p.root)
end

function MathOptAI.build_predictor(node::DecisionTree.Node{K,V}) where {K,V}
    return MathOptAI.BinaryDecisionTree{K,V}(
        node.featid,
        node.featval,
        MathOptAI.build_predictor(node.left),
        MathOptAI.build_predictor(node.right),
    )
end

MathOptAI.build_predictor(node::DecisionTree.Leaf) = node.majority

function MathOptAI.build_predictor(node::DecisionTree.Ensemble{K,V}) where {K,V}
    return MathOptAI.RandomForest{K,V}(
        [MathOptAI.build_predictor(t) for t in node.trees]
    )
end

end  # module
