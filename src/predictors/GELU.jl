# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    GeLU() <: AbstractPredictor

An [`AbstractPredictor`](@ref) representing the Gaussian Error Linear Units
function:
```math
y \\approx x * (1 + \\tanh(\\sqrt(2 / \\pi) * (x + 0.044715 x^3))) / 2
```
as a smooth nonlinear constraint.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, -1 <= x[i in 1:2] <= i);

julia> f = MathOptAI.GELU()
GELU()

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
2-element Vector{VariableRef}:
 moai_GELU[1]
 moai_GELU[2]

julia> formulation
GELU()
├ variables [2]
│ ├ moai_GELU[1]
│ └ moai_GELU[2]
└ constraints [6]
  ├ moai_GELU[1] ≥ -0.17
  ├ moai_GELU[1] ≤ 0.8411919906082768
  ├ moai_GELU[2] ≥ -0.17
  ├ moai_GELU[2] ≤ 1.954597694087775
  ├ moai_GELU[1] - ((0.5 x[1]) * (1.0 + tanh(0.7978845608028654 * (x[1] + (0.044715 * (x[1] ^ 3.0)))))) = 0
  └ moai_GELU[2] - ((0.5 x[2]) * (1.0 + tanh(0.7978845608028654 * (x[2] + (0.044715 * (x[2] ^ 3.0)))))) = 0
```
"""
struct GELU <: AbstractPredictor end

(::GELU)(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))

function add_predictor(model::JuMP.AbstractModel, predictor::GELU, x::Vector)
    y = add_variables(model, x, length(x), "moai_GELU")
    cons = Any[]
    for i in 1:length(x)
        x_l, x_u = get_variable_bounds(x[i])
        y_l = ismissing(x_l) ? -0.17 : (x_l >= 0 ? predictor(x_l) : -0.17)
        y_u = ismissing(x_u) ? missing : (x_u >= 0 ? predictor(x_u) : 0.0)
        set_variable_bounds(cons, y[i], y_l, y_u; optional = true)
    end
    append!(cons, JuMP.@constraint(model, y .== predictor.(x)))
    return y, Formulation(predictor, y, cons)
end

function add_predictor(
    ::JuMP.AbstractModel,
    predictor::ReducedSpace{GELU},
    x::Vector,
)
    return predictor.predictor.(x), Formulation(predictor)
end
