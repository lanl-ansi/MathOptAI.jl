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

julia> @variable(model, -1 <= x[i in 1:2] <= i);

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
  ├ moai_Sigmoid[1] ≥ 0.2689414213699951
  ├ moai_Sigmoid[1] ≤ 0.7310585786300049
  ├ moai_Sigmoid[2] ≥ 0.2689414213699951
  ├ moai_Sigmoid[2] ≤ 0.8807970779778823
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

_eval(::Sigmoid, x::Real) = 1 / (1 + exp(-x))

function add_predictor(model::JuMP.AbstractModel, predictor::Sigmoid, x::Vector)
    y = JuMP.@variable(model, [1:length(x)], base_name = "moai_Sigmoid")
    cons = Any[]
    for i in 1:length(x)
        x_l, x_u = _get_variable_bounds(x[i])
        y_l = x_l === nothing ? 0 : _eval(predictor, x_l)
        y_u = x_u === nothing ? 1 : _eval(predictor, x_u)
        _set_bounds_if_finite(cons, y[i], y_l, y_u)
    end
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
