# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    SoftPlus(; beta = 1.0) <: AbstractPredictor

An [`AbstractPredictor`](@ref) that implements the SoftPlus constraint
\$y = \\frac{1}{\\beta} \\log(1 + e^{\\beta x})\$ as a smooth nonlinear constraint.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, -1 <= x[i in 1:2] <= i);

julia> f = MathOptAI.SoftPlus(; beta = 2.0)
SoftPlus(2.0)

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
2-element Vector{VariableRef}:
 moai_SoftPlus[1]
 moai_SoftPlus[2]

julia> formulation
SoftPlus(2.0)
├ variables [2]
│ ├ moai_SoftPlus[1]
│ └ moai_SoftPlus[2]
└ constraints [6]
  ├ moai_SoftPlus[1] ≥ 0.0634640055214863
  ├ moai_SoftPlus[1] ≤ 1.0634640055214863
  ├ moai_SoftPlus[2] ≥ 0.0634640055214863
  ├ moai_SoftPlus[2] ≤ 2.0090749639589047
  ├ moai_SoftPlus[1] - (log(1.0 + exp(2 x[1])) / 2.0) = 0
  └ moai_SoftPlus[2] - (log(1.0 + exp(2 x[2])) / 2.0) = 0

julia> y, formulation =
           MathOptAI.add_predictor(model, MathOptAI.ReducedSpace(f), x);

julia> y
2-element Vector{NonlinearExpr}:
 log(1.0 + exp(2 x[1])) / 2.0
 log(1.0 + exp(2 x[2])) / 2.0

julia> formulation
ReducedSpace(SoftPlus(2.0))
├ variables [0]
└ constraints [0]
```
"""
struct SoftPlus <: AbstractPredictor
    beta::Float64
    SoftPlus(; beta::Float64 = 1.0) = new(beta)
end

_eval(f::SoftPlus, x::Real) = log(1 + exp(f.beta * x)) / f.beta

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::SoftPlus,
    x::Vector,
)
    y = JuMP.@variable(model, [1:length(x)], base_name = "moai_SoftPlus")
    cons = Any[]
    for i in 1:length(x)
        x_l, x_u = _get_variable_bounds(x[i])
        y_l = x_l === nothing ? 0 : _eval(predictor, x_l)
        y_u = x_u === nothing ? nothing : _eval(predictor, x_u)
        _set_bounds_if_finite(cons, y[i], y_l, y_u)
    end
    beta = predictor.beta
    append!(
        cons,
        JuMP.@constraint(model, y .== log.(1 .+ exp.(beta .* x)) ./ beta),
    )
    return y, Formulation(predictor, y, cons)
end

function add_predictor(
    ::JuMP.AbstractModel,
    predictor::ReducedSpace{SoftPlus},
    x::Vector,
)
    beta = predictor.predictor.beta
    return log.(1 .+ exp.(beta .* x)) ./ beta, Formulation(predictor)
end
