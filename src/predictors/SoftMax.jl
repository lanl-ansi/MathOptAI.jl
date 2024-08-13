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

julia> y = MathOptAI.add_predictor(model, MathOptAI.SoftMax(), x)
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
```
"""
struct SoftMax <: AbstractPredictor end

function add_predictor(
    model::JuMP.Model,
    ::SoftMax,
    x::Vector;
    reduced_space::Bool = false,
    kwargs...,
)
    denom = JuMP.@variable(model, base_name = "moai_SoftMax_denom")
    JuMP.set_lower_bound(denom, 0.0)
    JuMP.@constraint(model, denom == sum(exp.(x)))
    if reduced_space
        return exp.(x) ./ denom
    end
    y = JuMP.@variable(model, [1:length(x)], base_name = "moai_SoftMax")
    _set_bounds_if_finite.(y, 0.0, 1.0)
    JuMP.@constraint(model, y .== exp.(x) ./ denom)
    return y
end
