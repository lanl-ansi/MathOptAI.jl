# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    LinearCombination(
        predictors::Vector{<:AbstractPredictor},
        weights::Vector{Float64},
    )

An [`AbstractPredictor`](@ref) that represents the linear combination of other
predictors.

The main purpose of this predictor is to model random forests and gradient
boosted trees.

 * A random forest is the mean a set of [`BinaryDecisionTree`](@ref)
 * A gradient boosted tree is the sum of a set of [`BinaryDecisionTree`](@ref)

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> rhs = MathOptAI.BinaryDecisionTree(1, 1.0, 0, 1)
BinaryDecisionTree{Float64,Int64} [leaves=2, depth=2]

julia> lhs = MathOptAI.BinaryDecisionTree(1, -0.1, -1, 0)
BinaryDecisionTree{Float64,Int64} [leaves=2, depth=2]

julia> tree_1 = MathOptAI.BinaryDecisionTree(1, 0.0, -1, rhs);

julia> tree_2 = MathOptAI.BinaryDecisionTree(1, 0.9, lhs, 1);

julia> random_forest = MathOptAI.LinearCombination([tree_1, tree_2], [0.5, 0.5])
LinearCombination
├ 0.5 * BinaryDecisionTree{Float64,Int64} [leaves=3, depth=2]
└ 0.5 * BinaryDecisionTree{Float64,Int64} [leaves=3, depth=2]

julia> model = Model();

julia> @variable(model, -3 <= x[1:1] <= 5);

julia> y, formulation = MathOptAI.add_predictor(model, random_forest, x);

julia> y
1-element Vector{VariableRef}:
 moai_LinearCombination[1]

julia> formulation
LinearCombination
├ 0.5 * BinaryDecisionTree{Float64,Int64} [leaves=3, depth=2]
└ 0.5 * BinaryDecisionTree{Float64,Int64} [leaves=3, depth=2]
├ variables [1]
│ └ moai_LinearCombination[1]
└ constraints [1]
  └ 0.5 moai_BinaryDecisionTree_value + 0.5 moai_BinaryDecisionTree_value - moai_LinearCombination[1] = 0
BinaryDecisionTree{Float64,Int64} [leaves=3, depth=2]
├ variables [4]
│ ├ moai_BinaryDecisionTree_value
│ ├ moai_BinaryDecisionTree_z[1]
│ ├ moai_BinaryDecisionTree_z[2]
│ └ moai_BinaryDecisionTree_z[3]
└ constraints [7]
  ├ moai_BinaryDecisionTree_z[1] + moai_BinaryDecisionTree_z[2] + moai_BinaryDecisionTree_z[3] = 1
  ├ moai_BinaryDecisionTree_z[1] --> {x[1] ≤ 0}
  ├ moai_BinaryDecisionTree_z[2] --> {x[1] ≥ 0}
  ├ moai_BinaryDecisionTree_z[2] --> {x[1] ≤ 1}
  ├ moai_BinaryDecisionTree_z[3] --> {x[1] ≥ 0}
  ├ moai_BinaryDecisionTree_z[3] --> {x[1] ≥ 1}
  └ moai_BinaryDecisionTree_z[1] - moai_BinaryDecisionTree_z[3] + moai_BinaryDecisionTree_value = 0
BinaryDecisionTree{Float64,Int64} [leaves=3, depth=2]
├ variables [4]
│ ├ moai_BinaryDecisionTree_value
│ ├ moai_BinaryDecisionTree_z[1]
│ ├ moai_BinaryDecisionTree_z[2]
│ └ moai_BinaryDecisionTree_z[3]
└ constraints [7]
  ├ moai_BinaryDecisionTree_z[1] + moai_BinaryDecisionTree_z[2] + moai_BinaryDecisionTree_z[3] = 1
  ├ moai_BinaryDecisionTree_z[1] --> {x[1] ≤ 0.9}
  ├ moai_BinaryDecisionTree_z[1] --> {x[1] ≤ -0.1}
  ├ moai_BinaryDecisionTree_z[2] --> {x[1] ≤ 0.9}
  ├ moai_BinaryDecisionTree_z[2] --> {x[1] ≥ -0.1}
  ├ moai_BinaryDecisionTree_z[3] --> {x[1] ≥ 0.9}
  └ moai_BinaryDecisionTree_z[1] - moai_BinaryDecisionTree_z[3] + moai_BinaryDecisionTree_value = 0
```
"""
struct LinearCombination <: AbstractPredictor
    predictors::Vector{AbstractPredictor}
    weights::Vector{Float64}
end

function Base.show(io::IO, predictor::LinearCombination)
    println(io, "LinearCombination")
    for (i, (w, p)) in enumerate(zip(predictor.weights, predictor.predictors))
        if i == length(predictor.predictors)
            print(io, "└ ", w, " * ", p)
        else
            println(io, "├ ", w, " * ", p)
        end
    end
    return
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::LinearCombination,
    x::Vector;
    kwargs...,
)
    p = [add_predictor(model, p, x; kwargs...) for p in predictor.predictors]
    lhs = JuMP.@expression(
        model,
        sum(w * z for (w, (z, _)) in zip(predictor.weights, p)),
    )
    y = JuMP.@variable(
        model,
        [1:length(lhs)],
        base_name = "moai_LinearCombination",
    )
    c = JuMP.@constraint(model, lhs .== y)
    layers = vcat(Formulation(predictor, y, c), last.(p))
    return y, PipelineFormulation(predictor, layers)
end
