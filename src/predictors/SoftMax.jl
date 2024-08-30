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

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
2-element Vector{VariableRef}:
 moai_SoftMax[1]
 moai_SoftMax[2]

julia> formulation
SoftMax()
├ variables [3]
│ ├ moai_SoftMax_denom
│ ├ moai_SoftMax[1]
│ └ moai_SoftMax[2]
└ constraints [8]
  ├ moai_SoftMax[1] ≥ 0
  ├ moai_SoftMax[2] ≥ 0
  ├ moai_SoftMax[1] ≤ 1
  ├ moai_SoftMax[2] ≤ 1
  ├ moai_SoftMax_denom ≥ 0
  ├ moai_SoftMax_denom - (0.0 + exp(x[2]) + exp(x[1])) = 0
  ├ moai_SoftMax[1] - (exp(x[1]) / moai_SoftMax_denom) = 0
  └ moai_SoftMax[2] - (exp(x[2]) / moai_SoftMax_denom) = 0

julia> y, formulation =
           MathOptAI.add_predictor(model, MathOptAI.ReducedSpace(f), x);

julia> y
2-element Vector{NonlinearExpr}:
 exp(x[1]) / moai_SoftMax_denom
 exp(x[2]) / moai_SoftMax_denom

julia> formulation
ReducedSpace(SoftMax())
├ variables [1]
│ └ moai_SoftMax_denom
└ constraints [2]
  ├ moai_SoftMax_denom ≥ 0
  └ moai_SoftMax_denom - (0.0 + exp(x[2]) + exp(x[1])) = 0
```
"""
struct SoftMax <: AbstractPredictor end

function add_predictor(model::JuMP.AbstractModel, predictor::SoftMax, x::Vector)
    y = JuMP.@variable(model, [1:length(x)], base_name = "moai_SoftMax")
    _set_bounds_if_finite.(y, 0, 1)
    denom = JuMP.@variable(model, base_name = "moai_SoftMax_denom")
    JuMP.set_lower_bound(denom, 0)
    d_con = JuMP.@constraint(model, denom == sum(exp.(x)))
    cons = JuMP.@constraint(model, y .== exp.(x) ./ denom)
    constraints = [
        JuMP.LowerBoundRef.(y);
        JuMP.UpperBoundRef.(y);
        JuMP.LowerBoundRef(denom);
        d_con;
        cons;
    ]
    return y, Formulation(predictor, [denom; y], constraints)
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReducedSpace{SoftMax},
    x::Vector,
)
    denom = JuMP.@variable(model, base_name = "moai_SoftMax_denom")
    JuMP.set_lower_bound(denom, 0)
    d_con = JuMP.@constraint(model, denom == sum(exp.(x)))
    constraints = Any[JuMP.LowerBoundRef(denom); d_con]
    return exp.(x) ./ denom, Formulation(predictor, [denom], constraints)
end
