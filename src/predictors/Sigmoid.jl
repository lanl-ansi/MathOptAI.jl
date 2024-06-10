# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    Sigmoid() <: AbstractPredictor

An [`AbstractPredictor`](@ref) that implements the Sigmoid constraint
\$y = \\frac{1}{1 + e^{-x}}\$ as a smooth nonlinear constraint.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> y = MathOptAI.add_predictor(model, MathOptAI.Sigmoid(), x)
2-element Vector{VariableRef}:
 omelette_Sigmoid[1]
 omelette_Sigmoid[2]

julia> print(model)
Feasibility
Subject to
 omelette_Sigmoid[1] - (1.0 / (1.0 + exp(-x[1]))) = 0
 omelette_Sigmoid[2] - (1.0 / (1.0 + exp(-x[2]))) = 0
 omelette_Sigmoid[1] ≥ 0
 omelette_Sigmoid[2] ≥ 0
 omelette_Sigmoid[1] ≤ 1
 omelette_Sigmoid[2] ≤ 1
```
"""
struct Sigmoid <: AbstractPredictor end

function add_predictor(model::JuMP.Model, predictor::Sigmoid, x::Vector)
    y = JuMP.@variable(model, [1:length(x)], base_name = "omelette_Sigmoid")
    _set_bounds_if_finite.(y, 0.0, 1.0)
    JuMP.@constraint(model, [i in 1:length(x)], y[i] == 1 / (1 + exp(-x[i])))
    return y
end
