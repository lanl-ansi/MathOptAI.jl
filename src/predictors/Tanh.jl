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

julia> f = MathOptAI.Tanh()
Tanh()

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
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

julia> y, formulation = MathOptAI.add_predictor(model, MathOptAI.ReducedSpace(f), x);

julia> y
2-element Vector{NonlinearExpr}:
 tanh(x[1])
 tanh(x[2])
```
"""
struct Tanh <: AbstractPredictor end

function add_predictor(model::JuMP.AbstractModel, predictor::Tanh, x::Vector)
    y = JuMP.@variable(model, [1:length(x)], base_name = "moai_Tanh")
    _set_bounds_if_finite.(y, -1, 1)
    cons = JuMP.@constraint(model, y .== tanh.(x))
    return y, SimpleFormulation(predictor, y, cons)
end

function add_predictor(
    ::JuMP.AbstractModel,
    predictor::ReducedSpace{Tanh},
    x::Vector,
)
    return tanh.(x), SimpleFormulation(predictor)
end
