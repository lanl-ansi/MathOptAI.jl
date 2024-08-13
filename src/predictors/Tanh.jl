# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    Tanh() <: AbstractPredictor

An [`AbstractPredictor`](@ref) that implements the Tanh constraint
\$y = \\tanh(x)\$ as a smooth nonlinear constraint.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> y = MathOptAI.add_predictor(model, MathOptAI.Tanh(), x)
2-element Vector{VariableRef}:
 moai_Tanh[1]
 moai_Tanh[2]

julia> print(model)
Feasibility
Subject to
 moai_Tanh[1] - tanh(x[1]) = 0
 moai_Tanh[2] - tanh(x[2]) = 0
 moai_Tanh[1] ≥ -1
 moai_Tanh[2] ≥ -1
 moai_Tanh[1] ≤ 1
 moai_Tanh[2] ≤ 1
```
"""
struct Tanh <: AbstractPredictor end

function add_predictor(
    model::JuMP.Model,
    ::Tanh,
    x::Vector;
    reduced_space::Bool = false,
    kwargs...,
)
    if reduced_space
        return tanh.(x)
    end
    y = JuMP.@variable(model, [1:length(x)], base_name = "moai_Tanh")
    _set_bounds_if_finite.(y, -1.0, 1.0)
    JuMP.@constraint(model, y .== tanh.(x))
    return y
end
