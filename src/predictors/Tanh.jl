# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    Tanh()

Implements the Tanh constraint \$y = tanh(x)\$ directly.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> y = MathOptAI.add_predictor(model, MathOptAI.Tanh(), x)
2-element Vector{VariableRef}:
 omelette_Tanh[1]
 omelette_Tanh[2]

julia> print(model)
Feasibility
Subject to
 omelette_Tanh[1] - tanh(x[1]) = 0
 omelette_Tanh[2] - tanh(x[2]) = 0
 omelette_Tanh[1] ≥ -1
 omelette_Tanh[2] ≥ -1
 omelette_Tanh[1] ≤ 1
 omelette_Tanh[2] ≤ 1
```
"""
struct Tanh <: AbstractPredictor end

function add_predictor(model::JuMP.Model, predictor::Tanh, x::Vector)
    y = JuMP.@variable(model, [1:length(x)], base_name = "omelette_Tanh")
    _set_bounds_if_finite.(y, -1.0, 1.0)
    JuMP.@constraint(model, y .== tanh.(x))
    return y
end
