# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    SoftPlusEpigraph(; beta = 1.0) <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents the relationship:
```math
y \\ge \\frac{1}{\\beta} \\log(1 + e^{\\beta x})
```
as a smooth nonlinear constraint.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, -1 <= x[i in 1:2] <= i);

julia> f = MathOptAI.SoftPlusEpigraph(; beta = 2.0)
SoftPlusEpigraph(2.0)

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
2-element Vector{VariableRef}:
 moai_SoftPlusEpigraph[1]
 moai_SoftPlusEpigraph[2]

julia> formulation
SoftPlusEpigraph(2.0)
├ variables [2]
│ ├ moai_SoftPlusEpigraph[1]
│ └ moai_SoftPlusEpigraph[2]
└ constraints [6]
  ├ moai_SoftPlusEpigraph[1] ≥ 0
  ├ moai_SoftPlusEpigraph[1] ≤ 1
  ├ moai_SoftPlusEpigraph[1] - (log(1 + exp(2 x[1])) / 2) ≥ 0
  ├ moai_SoftPlusEpigraph[2] ≥ 0
  ├ moai_SoftPlusEpigraph[2] ≤ 2
  └ moai_SoftPlusEpigraph[2] - (log(1 + exp(2 x[2])) / 2) ≥ 0
```
"""
struct SoftPlusEpigraph <: MathOptAI.AbstractPredictor
    beta::Float64
    SoftPlusEpigraph(; beta::Float64 = 1.0) = new(beta)
end

MathOptAI.output_size(::SoftPlusEpigraph, input_size) = input_size

function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
    predictor::SoftPlusEpigraph,
    x::Vector,
)
    sp = MathOptAI.SoftPlus(; beta = predictor.beta)
    y = add_variables(model, x, length(x), "moai_SoftPlusEpigraph")
    cons = Any[]
    for i in 1:length(x)
        l, u = max.(0, get_variable_bounds(x[i]))
        set_variable_bounds(cons, y[i], coalesce(l, 0), u; optional = false)
        push!(cons, JuMP.@constraint(model, y[i] >= sp(x[i])))
    end
    return y, Formulation(predictor, y, cons)
end
