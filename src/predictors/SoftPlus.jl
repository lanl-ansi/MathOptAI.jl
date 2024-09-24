# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    SoftPlus(beta::T = 1.0) where {T} <: AbstractPredictor

An [`AbstractPredictor`](@ref) that implements the SoftPlus constraint
\$y = \\frac{1}{\\beta} \\log(1 + e^{\\beta x})\$ as a smooth nonlinear constraint.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = MathOptAI.SoftPlus(2.0)
SoftPlus{Float64}(2.0)

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
2-element Vector{VariableRef}:
 moai_SoftPlus[1]
 moai_SoftPlus[2]

julia> formulation
SoftPlus{Float64}(2.0)
├ variables [2]
│ ├ moai_SoftPlus[1]
│ └ moai_SoftPlus[2]
└ constraints [4]
  ├ moai_SoftPlus[1] ≥ 0
  ├ moai_SoftPlus[2] ≥ 0
  ├ moai_SoftPlus[1] - (log(1.0 + exp(2 x[1])) / 2.0) = 0
  └ moai_SoftPlus[2] - (log(1.0 + exp(2 x[2])) / 2.0) = 0

julia> y, formulation =
           MathOptAI.add_predictor(model, MathOptAI.ReducedSpace(f), x);

julia> y
2-element Vector{NonlinearExpr}:
 log(1.0 + exp(2 x[1])) / 2.0
 log(1.0 + exp(2 x[2])) / 2.0

julia> formulation
ReducedSpace(SoftPlus{Float64}(2.0))
├ variables [0]
└ constraints [0]
```
"""
struct SoftPlus{T} <: AbstractPredictor
    beta::T
end

SoftPlus() = SoftPlus(1.0)

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::SoftPlus{T},
    x::Vector,
) where T
    y = JuMP.@variable(model, [1:length(x)], base_name = "moai_SoftPlus")
    _set_bounds_if_finite.(y, 0, nothing)
    beta = predictor.beta
    cons = JuMP.@constraint(model, y .== log.(1 .+ exp.(beta .* x)) ./ beta)
    return y, Formulation(predictor, y, Any[JuMP.LowerBoundRef.(y); cons])
end

function add_predictor(
    ::JuMP.AbstractModel,
    predictor::ReducedSpace{SoftPlus{T}},
    x::Vector,
) where T
    beta = predictor.predictor.beta
    return log.(1 .+ exp.(beta .* x)) ./ beta, Formulation(predictor)
end
