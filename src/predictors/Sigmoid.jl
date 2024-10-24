# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
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

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
2-element Vector{VariableRef}:
 moai_Sigmoid[1]
 moai_Sigmoid[2]

julia> formulation
Sigmoid()
├ variables [2]
│ ├ moai_Sigmoid[1]
│ └ moai_Sigmoid[2]
└ constraints [6]
  ├ moai_Sigmoid[1] ≥ 0
  ├ moai_Sigmoid[1] ≤ 1
  ├ moai_Sigmoid[2] ≥ 0
  ├ moai_Sigmoid[2] ≤ 1
  ├ moai_Sigmoid[1] - (1.0 / (1.0 + exp(-x[1]))) = 0
  └ moai_Sigmoid[2] - (1.0 / (1.0 + exp(-x[2]))) = 0

julia> y, formulation =
           MathOptAI.add_predictor(model, MathOptAI.ReducedSpace(f), x);

julia> y
2-element Vector{NonlinearExpr}:
 1.0 / (1.0 + exp(-x[1]))
 1.0 / (1.0 + exp(-x[2]))

julia> formulation
ReducedSpace(Sigmoid())
├ variables [0]
└ constraints [0]
```
"""
struct Sigmoid <: AbstractPredictor end

function add_predictor(model::JuMP.AbstractModel, predictor::Sigmoid, x::Vector)
    y = JuMP.@variable(model, [1:length(x)], base_name = "moai_Sigmoid")
    cons = Any[]
    _set_bounds_if_finite.(Ref(cons), y, 0, 1)
    append!(cons, JuMP.@constraint(model, y .== 1 ./ (1 .+ exp.(-x))))
    return y, Formulation(predictor, y, cons)
end

function add_predictor(
    ::JuMP.AbstractModel,
    predictor::ReducedSpace{Sigmoid},
    x::Vector,
)
    return 1 ./ (1 .+ exp.(-x)), Formulation(predictor)
end
