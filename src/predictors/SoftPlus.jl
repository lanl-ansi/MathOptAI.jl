# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    SoftPlus() <: AbstractPredictor

An [`AbstractPredictor`](@ref) that implements the SoftPlus constraint
\$y = \\log(1 + e^x)\$ as a smooth nonlinear constraint.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = MathOptAI.SoftPlus()
SoftPlus()

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
2-element Vector{VariableRef}:
 moai_SoftPlus[1]
 moai_SoftPlus[2]

julia> formulation
SoftPlus()
├ variables [2]
│ ├ moai_SoftPlus[1]
│ └ moai_SoftPlus[2]
└ constraints [2]
  ├ moai_SoftPlus[1] - log(1.0 + exp(x[1])) = 0
  └ moai_SoftPlus[2] - log(1.0 + exp(x[2])) = 0

julia> y, formulation =
           MathOptAI.add_predictor(model, MathOptAI.ReducedSpace(f), x);

julia> y
2-element Vector{NonlinearExpr}:
 log(1.0 + exp(x[1]))
 log(1.0 + exp(x[2]))

julia> formulation
ReducedSpace{SoftPlus}(SoftPlus())
├ variables [0]
└ constraints [0]
```
"""
struct SoftPlus <: AbstractPredictor end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::SoftPlus,
    x::Vector,
)
    y = JuMP.@variable(model, [1:length(x)], base_name = "moai_SoftPlus")
    _set_bounds_if_finite.(y, 0, nothing)
    cons = JuMP.@constraint(model, y .== log.(1 .+ exp.(x)))
    return y, SimpleFormulation(predictor, y, cons)
end

function add_predictor(
    ::JuMP.AbstractModel,
    predictor::ReducedSpace{SoftPlus},
    x::Vector,
)
    return log.(1 .+ exp.(x)), SimpleFormulation(predictor)
end
