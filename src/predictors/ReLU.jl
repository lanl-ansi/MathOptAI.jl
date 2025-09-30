# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    ReLU() <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents the relationship:
```math
y = \\max\\{0, x\\}
```
as a non-smooth nonlinear constraint.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, -1 <= x[i in 1:2] <= i);

julia> f = MathOptAI.ReLU()
ReLU()

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
2-element Vector{VariableRef}:
 moai_ReLU[1]
 moai_ReLU[2]

julia> formulation
ReLU()
├ variables [2]
│ ├ moai_ReLU[1]
│ └ moai_ReLU[2]
└ constraints [6]
  ├ moai_ReLU[1] ≥ 0
  ├ moai_ReLU[1] ≤ 1
  ├ moai_ReLU[1] - max(0.0, x[1]) = 0
  ├ moai_ReLU[2] ≥ 0
  ├ moai_ReLU[2] ≤ 2
  └ moai_ReLU[2] - max(0.0, x[2]) = 0

julia> y, formulation =
           MathOptAI.add_predictor(model, MathOptAI.ReducedSpace(f), x);

julia> y
2-element Vector{NonlinearExpr}:
 max(0.0, x[1])
 max(0.0, x[2])

julia> formulation
ReducedSpace(ReLU())
├ variables [0]
└ constraints [0]
```
"""
struct ReLU <: AbstractPredictor end

function add_predictor(model::JuMP.AbstractModel, predictor::ReLU, x::Vector)
    y = add_variables(model, x, length(x), "moai_ReLU")
    cons = Any[]
    for i in 1:length(x)
        l, u = max.(0, get_variable_bounds(x[i]))
        set_variable_bounds(cons, y[i], coalesce(l, 0), u; optional = true)
        push!(cons, JuMP.@constraint(model, y[i] == max(0, x[i])))
    end
    return y, Formulation(predictor, y, cons)
end

function add_predictor(
    ::JuMP.AbstractModel,
    predictor::ReducedSpace{ReLU},
    x::Vector,
)
    return max.(0, x), Formulation(predictor)
end
