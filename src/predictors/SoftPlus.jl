# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    SoftPlus(; beta = 1.0) <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents the relationship:
```math
y = \\frac{1}{\\beta} \\log(1 + e^{\\beta x})
```
as a smooth nonlinear constraint.

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
  ├ moai_SoftPlus[1] - (log(1.0 + exp(2 x[1])) / 2.0) = 0
  ├ moai_SoftPlus[2] ≥ 0.0634640055214863
  ├ moai_SoftPlus[2] ≤ 2.0090749639589047
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

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::SoftPlus,
    x::Vector,
)
    y = add_variables(model, predictor, x, length(x), "moai_SoftPlus")
    cons = Any[]
    f(x) = log(1 + exp(predictor.beta * x)) / predictor.beta
    for i in 1:length(x)
        l, u = f.(get_variable_bounds(x[i]))
        set_variable_bounds(cons, y[i], coalesce(l, 0), u; optional = true)
        push!(cons, JuMP.@constraint(model, y[i] == f(x[i])))
    end
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
