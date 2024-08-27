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

julia> f = MathOptAI.Sigmoid()
Sigmoid()

julia> y = MathOptAI.add_predictor(model, f, x)
2-element Vector{VariableRef}:
 moai_Sigmoid[1]
 moai_Sigmoid[2]

julia> print(model)
Feasibility
Subject to
 moai_Sigmoid[1] - (1.0 / (1.0 + exp(-x[1]))) = 0
 moai_Sigmoid[2] - (1.0 / (1.0 + exp(-x[2]))) = 0
 moai_Sigmoid[1] ≥ 0
 moai_Sigmoid[2] ≥ 0
 moai_Sigmoid[1] ≤ 1
 moai_Sigmoid[2] ≤ 1

julia> y = MathOptAI.add_predictor(model, MathOptAI.ReducedSpace(f), x)
2-element Vector{NonlinearExpr}:
 1.0 / (1.0 + exp(-x[1]))
 1.0 / (1.0 + exp(-x[2]))
```
"""
struct Sigmoid <: AbstractPredictor end

function add_predictor(model::JuMP.AbstractModel, ::Sigmoid, x::Vector)
    y = JuMP.@variable(model, [1:length(x)], base_name = "moai_Sigmoid")
    _set_bounds_if_finite.(y, 0.0, 1.0)
    JuMP.@constraint(model, [i in 1:length(x)], y[i] == 1 / (1 + exp(-x[i])))
    return y
end

function add_predictor(::JuMP.AbstractModel, ::ReducedSpace{Sigmoid}, x::Vector)
    return 1 ./ (1 .+ exp.(-x))
end
