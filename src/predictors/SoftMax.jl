# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    SoftMax() <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents the relationship:
```math
y = \\frac{e^{x}}{||e^{x}||_1}
```
as a smooth nonlinear constraint.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = MathOptAI.SoftMax()
SoftMax()

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
2-element Vector{VariableRef}:
 moai_SoftMax[1]
 moai_SoftMax[2]

julia> formulation
SoftMax()
├ variables [3]
│ ├ moai_SoftMax_denom[1]
│ ├ moai_SoftMax[1]
│ └ moai_SoftMax[2]
└ constraints [8]
  ├ moai_SoftMax_denom[1] ≥ 0
  ├ moai_SoftMax_denom[1] - (0.0 + exp(x[2]) + exp(x[1])) = 0
  ├ moai_SoftMax[1] ≥ 0
  ├ moai_SoftMax[1] ≤ 1
  ├ moai_SoftMax[1] - (exp(x[1]) / moai_SoftMax_denom[1]) = 0
  ├ moai_SoftMax[2] ≥ 0
  ├ moai_SoftMax[2] ≤ 1
  └ moai_SoftMax[2] - (exp(x[2]) / moai_SoftMax_denom[1]) = 0

julia> y, formulation =
           MathOptAI.add_predictor(model, MathOptAI.ReducedSpace(f), x);

julia> y
2-element Vector{NonlinearExpr}:
 exp(x[1]) / moai_SoftMax_denom[1]
 exp(x[2]) / moai_SoftMax_denom[1]

julia> formulation
ReducedSpace(SoftMax())
├ variables [1]
│ └ moai_SoftMax_denom[1]
└ constraints [2]
  ├ moai_SoftMax_denom[1] ≥ 0
  └ moai_SoftMax_denom[1] - (0.0 + exp(x[2]) + exp(x[1])) = 0
```
"""
struct SoftMax <: AbstractPredictor end

function add_predictor(model::JuMP.AbstractModel, predictor::SoftMax, x::Vector)
    cons = Any[]
    y = add_variables(model, predictor, x, length(x), "moai_SoftMax")
    denom = only(add_variables(model, predictor, x, 1, "moai_SoftMax_denom"))
    set_variable_bounds(cons, denom, 0, missing; optional = true)
    push!(cons, JuMP.@constraint(model, denom == sum(exp.(x))))
    for i in 1:length(x)
        set_variable_bounds(cons, y[i], 0, 1; optional = true)
        push!(cons, JuMP.@constraint(model, y[i] == exp(x[i]) / denom))
    end
    return y, Formulation(predictor, [denom; y], cons)
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReducedSpace{SoftMax},
    x::Vector,
)
    cons = Any[]
    denom = only(add_variables(model, predictor, x, 1, "moai_SoftMax_denom"))
    set_variable_bounds(cons, denom, 0, missing; optional = true)
    push!(cons, JuMP.@constraint(model, denom == sum(exp.(x))))
    return exp.(x) ./ denom, Formulation(predictor, [denom], cons)
end
