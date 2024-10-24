# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
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

julia> @variable(model, -1 <= x[i in 1:2] <= i);

julia> f = MathOptAI.Tanh()
Tanh()

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
2-element Vector{VariableRef}:
 moai_Tanh[1]
 moai_Tanh[2]

julia> formulation
Tanh()
├ variables [2]
│ ├ moai_Tanh[1]
│ └ moai_Tanh[2]
└ constraints [6]
  ├ moai_Tanh[1] ≥ -0.7615941559557649
  ├ moai_Tanh[1] ≤ 0.7615941559557649
  ├ moai_Tanh[2] ≥ -0.7615941559557649
  ├ moai_Tanh[2] ≤ 0.9640275800758169
  ├ moai_Tanh[1] - tanh(x[1]) = 0
  └ moai_Tanh[2] - tanh(x[2]) = 0

julia> y, formulation =
           MathOptAI.add_predictor(model, MathOptAI.ReducedSpace(f), x);

julia> y
2-element Vector{NonlinearExpr}:
 tanh(x[1])
 tanh(x[2])

julia> formulation
ReducedSpace(Tanh())
├ variables [0]
└ constraints [0]
```
"""
struct Tanh <: AbstractPredictor end

function add_predictor(model::JuMP.AbstractModel, predictor::Tanh, x::Vector)
    y = JuMP.@variable(model, [1:length(x)], base_name = "moai_Tanh")
    cons = _set_variable_bounds(tanh, -1, 1, x, y)
    append!(cons, JuMP.@constraint(model, y .== tanh.(x)))
    return y, Formulation(predictor, y, cons)
end

function add_predictor(
    ::JuMP.AbstractModel,
    predictor::ReducedSpace{Tanh},
    x::Vector,
)
    return tanh.(x), Formulation(predictor)
end
