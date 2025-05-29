# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    RandomForest{K,V}(trees::Vector{BinaryDecisionTree{K,V}})

An [`AbstractPredictor`](@ref) that represents a random forest.

A random forest is the average of a set of [`BinaryDecisionTree`](@ref).

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> rhs = MathOptAI.BinaryDecisionTree(1, 1.0, 0, 1)
BinaryDecisionTree{Float64,Int64} [leaves=2, depth=2]

julia> lhs = MathOptAI.BinaryDecisionTree(1, -0.1, -1, 0)
BinaryDecisionTree{Float64,Int64} [leaves=2, depth=2]

julia> predictor = MathOptAI.RandomForest([
           MathOptAI.BinaryDecisionTree(1, 0.0, -1, rhs),
           MathOptAI.BinaryDecisionTree(1, 0.9, lhs, 1),
       ])
RandomForest{Float64,Int64} [trees=2]

julia> model = Model();

julia> @variable(model, -3 <= x[1:1] <= 5);

julia> y, formulation = MathOptAI.add_predictor(model, predictor, x);

julia> y
1-element Vector{VariableRef}:
 moai_RandomForest

julia> formulation
RandomForest{Float64,Int64} [trees=2]
├ variables [9]
│ ├ moai_RandomForest
│ ├ moai_BinaryDecisionTree_value
│ ├ moai_BinaryDecisionTree_z[1]
│ ├ moai_BinaryDecisionTree_z[2]
│ ├ moai_BinaryDecisionTree_z[3]
│ ├ moai_BinaryDecisionTree_value
│ ├ moai_BinaryDecisionTree_z[1]
│ ├ moai_BinaryDecisionTree_z[2]
│ └ moai_BinaryDecisionTree_z[3]
└ constraints [15]
  ├ moai_BinaryDecisionTree_value + moai_BinaryDecisionTree_value - 2 moai_RandomForest = 0
  ├ moai_BinaryDecisionTree_z[1] + moai_BinaryDecisionTree_z[2] + moai_BinaryDecisionTree_z[3] = 1
  ├ moai_BinaryDecisionTree_z[1] --> {x[1] ≤ 0}
  ├ moai_BinaryDecisionTree_z[2] --> {x[1] ≥ 0}
  ├ moai_BinaryDecisionTree_z[2] --> {x[1] ≤ 1}
  ├ moai_BinaryDecisionTree_z[3] --> {x[1] ≥ 0}
  ├ moai_BinaryDecisionTree_z[3] --> {x[1] ≥ 1}
  ├ moai_BinaryDecisionTree_z[1] - moai_BinaryDecisionTree_z[3] + moai_BinaryDecisionTree_value = 0
  ├ moai_BinaryDecisionTree_z[1] + moai_BinaryDecisionTree_z[2] + moai_BinaryDecisionTree_z[3] = 1
  ├ moai_BinaryDecisionTree_z[1] --> {x[1] ≤ 0.9}
  ├ moai_BinaryDecisionTree_z[1] --> {x[1] ≤ -0.1}
  ├ moai_BinaryDecisionTree_z[2] --> {x[1] ≤ 0.9}
  ├ moai_BinaryDecisionTree_z[2] --> {x[1] ≥ -0.1}
  ├ moai_BinaryDecisionTree_z[3] --> {x[1] ≥ 0.9}
  └ moai_BinaryDecisionTree_z[1] - moai_BinaryDecisionTree_z[3] + moai_BinaryDecisionTree_value = 0
```
"""
struct RandomForest{K,V} <: AbstractPredictor
    trees::Vector{BinaryDecisionTree{K,V}}
end

function Base.show(io::IO, predictor::RandomForest{K,V}) where {K,V}
    n = length(predictor.trees)
    return print(io, "RandomForest{$K,$V} [trees=$n]")
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::RandomForest,
    x::Vector;
    atol::Float64 = 0.0,
)
    trees = [add_predictor(model, p, x; atol) for p in predictor.trees]
    y = JuMP.@variable(model, base_name = "moai_RandomForest")
    c = JuMP.@constraint(
        model,
        sum(only(z) for (z, _) in trees) == length(trees) * y,
    )
    variables = reduce(vcat, f.variables for (_, f) in trees)
    constraints = reduce(vcat, f.constraints for (_, f) in trees)
    formulation = Formulation(predictor, [y; variables], [c; constraints])
    return [y], formulation
end
