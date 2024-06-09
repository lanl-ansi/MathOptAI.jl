# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    SoftPlus()

Implements the SoftPlus constraint \$y = log(1 + e^x)\$ directly.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> y = MathOptAI.add_predictor(model, MathOptAI.SoftPlus(), x)
2-element Vector{VariableRef}:
 omelette_SoftPlus[1]
 omelette_SoftPlus[2]

julia> print(model)
Feasibility
Subject to
 omelette_SoftPlus[1] - log(1.0 + exp(x[1])) = 0
 omelette_SoftPlus[2] - log(1.0 + exp(x[2])) = 0
 omelette_SoftPlus[1] ≥ 0
 omelette_SoftPlus[2] ≥ 0
```
"""
struct SoftPlus <: AbstractPredictor end

function add_predictor(model::JuMP.Model, predictor::SoftPlus, x::Vector)
    y = JuMP.@variable(model, [1:length(x)], base_name = "omelette_SoftPlus")
    _set_bounds_if_finite.(y, 0.0, Inf)
    JuMP.@constraint(model, y .== log.(1 .+ exp.(x)))
    return y
end
