# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAIEvoTreesExt

import EvoTrees
import JuMP
import MathOptAI

"""
    MathOptAI.add_predictor(
        model::JuMP.AbstractModel,
        predictor::EvoTrees.EvoTree,
        x::Vector,
    )

Add a boosted tree from EvoTrees.jl to `model`.

## Example

```jldoctest
julia> using JuMP, MathOptAI, EvoTrees

julia> truth(x::Vector) = x[1] <= 0.5 ? -2 : (x[2] <= 0.3 ? 3 : 4)
truth (generic function with 1 method)

julia> x_train = abs.(sin.((1:10) .* (3:4)'));

julia> size(x_train)
(10, 2)

julia> y_train = truth.(Vector.(eachrow(x_train)));

julia> config = EvoTrees.EvoTreeRegressor(; nrounds = 3);

julia> predictor = EvoTrees.fit(config; x_train, y_train);

julia> model = Model();

julia> @variable(model, 0 <= x[1:2] <= 1);

julia> y, _ = MathOptAI.add_predictor(model, predictor, x);

julia> y
1-element Vector{VariableRef}:
 moai_AffineCombination[1]
```
"""
function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
    predictor::EvoTrees.EvoTree{L,1},
    x::Vector,
) where {L}
    inner_predictor = MathOptAI.build_predictor(predictor)
    return MathOptAI.add_predictor(model, inner_predictor, x)
end

"""
    MathOptAI.build_predictor(predictor::DecisionTree.Root)

Convert a boosted tree from EvoTrees.jl to a [`AffineCombination`](@ref) of
[`BinaryDecisionTree`](@ref).

## Example

```jldoctest
julia> using MathOptAI, EvoTrees

julia> truth(x::Vector) = x[1] <= 0.5 ? -2 : (x[2] <= 0.3 ? 3 : 4)
truth (generic function with 1 method)

julia> x_train = abs.(sin.((1:10) .* (3:4)'));

julia> size(x_train)
(10, 2)

julia> y_train = truth.(Vector.(eachrow(x_train)));

julia> config = EvoTrees.EvoTreeRegressor(; nrounds = 3);

julia> tree = EvoTrees.fit(config; x_train, y_train);

julia> predictor = MathOptAI.build_predictor(tree)
AffineCombination
├ 1.0 * BinaryDecisionTree{Float64,Float64} [leaves=3, depth=2]
├ 1.0 * BinaryDecisionTree{Float64,Float64} [leaves=3, depth=2]
├ 1.0 * BinaryDecisionTree{Float64,Float64} [leaves=3, depth=2]
└ 1.0 * [2.0]
```
"""
function _to_tree(predictor::EvoTrees.EvoTree, tree::EvoTrees.Tree, i::Int = 1)
    if iszero(tree.feat[i])  # It's a leaf
        return Float64(tree.pred[i])
    end
    return MathOptAI.BinaryDecisionTree{Float64,Float64}(
        tree.feat[i],
        predictor.info[:edges][tree.feat[i]][tree.cond_bin[i]],
        _to_tree(predictor, tree, i << 1),
        _to_tree(predictor, tree, i << 1 + true),
    )
end

function MathOptAI.build_predictor(predictor::EvoTrees.EvoTree{L,1}) where {L}
    trees = MathOptAI.AbstractPredictor[]
    constant = 0.0
    for tree in predictor.trees
        p = _to_tree(predictor, tree)
        if p isa MathOptAI.BinaryDecisionTree
            push!(trees, p)
        else
            constant += p
        end
    end
    return MathOptAI.AffineCombination(trees, ones(length(trees)), [constant])
end

end  # module
