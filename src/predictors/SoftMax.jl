# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    SoftMax() <: AbstractPredictor

An [`AbstractPredictor`](@ref) that implements the SoftMax constraint
\$y_i = \\frac{e^{x_i}}{\\sum_j e^{x_j}}\$ as a smooth nonlinear constraint.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = MathOptAI.SoftMax()
SoftMax()

julia> y = MathOptAI.add_predictor(model, f, x)
2-element Vector{VariableRef}:
 moai_SoftMax[1]
 moai_SoftMax[2]

julia> print(model)
Feasibility
Subject to
 moai_SoftMax_denom - (0.0 + exp(x[2]) + exp(x[1])) = 0
 moai_SoftMax[1] - (exp(x[1]) / moai_SoftMax_denom) = 0
 moai_SoftMax[2] - (exp(x[2]) / moai_SoftMax_denom) = 0
 moai_SoftMax[1] ≥ 0
 moai_SoftMax[2] ≥ 0
 moai_SoftMax_denom ≥ 0
 moai_SoftMax[1] ≤ 1
 moai_SoftMax[2] ≤ 1

julia> y = MathOptAI.add_predictor(model, MathOptAI.ReducedSpace(f), x)
2-element Vector{NonlinearExpr}:
 exp(x[1]) / moai_SoftMax_denom
 exp(x[2]) / moai_SoftMax_denom
```
"""
struct SoftMax <: AbstractPredictor end

function add_predictor(model::JuMP.AbstractModel, predictor::SoftMax, x::Vector)
    vars = add_variables(
        model,
        predictor,
        x,
        1 + length(x);
        base_name = "moai_SoftMax",
    )
    denom, y = vars[1], vars[2:end]
    set_bounds.(y, 0, 1)
    JuMP.set_name(denom, "moai_SoftMax_denom")
    set_bounds(denom, 0, nothing)
    JuMP.@constraint(model, denom == sum(exp.(x)))
    JuMP.@constraint(model, y .== exp.(x) ./ denom)
    return y
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReducedSpace{SoftMax},
    x::Vector,
)
    vars =
        add_variables(model, predictor, x, 1; base_name = "moai_SoftMax_denom")
    denom = only(vars)
    set_bounds(denom, 0, nothing)
    JuMP.@constraint(model, denom == sum(exp.(x)))
    return exp.(x) ./ denom
end
