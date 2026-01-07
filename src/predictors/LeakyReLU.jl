# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    LeakyReLU(;
        negative_slope::Float64,
        relu::AbstractPredictor = ReLU(),
    ) <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents the relationship:
```math
y = \\max\\{0, x\\} + \\eta \\cdot \\min\\{0, x\\}
```
or equivalently:
```math
y = \\eta \\cdot x + (1 - \\eta) \\cdot \\max\\{0, x\\}
```
where `negative_slope` is \$\\eta\$ and `relu` is used to represent
\$\\max\\{0, x\\}\$.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, -1 <= x[i in 1:2] <= i);

julia> f = MathOptAI.LeakyReLU(; negative_slope = 0.01)
LeakyReLU{ReLU}(0.01, ReLU())

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
2-element Vector{VariableRef}:
 moai_LeakyReLU[1]
 moai_LeakyReLU[2]

julia> formulation
LeakyReLU{ReLU}(0.01, ReLU())
├ variables [4]
│ ├ moai_ReLU[1]
│ ├ moai_ReLU[2]
│ ├ moai_LeakyReLU[1]
│ └ moai_LeakyReLU[2]
└ constraints [8]
  ├ moai_ReLU[1] ≥ 0
  ├ moai_ReLU[1] ≤ 1
  ├ moai_ReLU[1] - max(0.0, x[1]) = 0
  ├ moai_ReLU[2] ≥ 0
  ├ moai_ReLU[2] ≤ 2
  ├ moai_ReLU[2] - max(0.0, x[2]) = 0
  ├ -0.01 x[1] + moai_LeakyReLU[1] - 0.99 moai_ReLU[1] = 0
  └ -0.01 x[2] + moai_LeakyReLU[2] - 0.99 moai_ReLU[2] = 0

julia> y, formulation =
           MathOptAI.add_predictor(model, MathOptAI.ReducedSpace(f), x);

julia> y
2-element Vector{NonlinearExpr}:
 (0.01 x[1]) + (0.99 * max(0.0, x[1]))
 (0.01 x[2]) + (0.99 * max(0.0, x[2]))

julia> formulation
ReducedSpace(LeakyReLU{ReLU}(0.01, ReLU()))
├ variables [0]
└ constraints [0]
```
"""
struct LeakyReLU{P} <: AbstractPredictor
    negative_slope::Float64
    relu::P

    function LeakyReLU(;
        negative_slope::Float64,
        relu::AbstractPredictor = ReLU(),
    )
        @assert negative_slope > 0.0
        return new{typeof(relu)}(negative_slope, relu)
    end
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::LeakyReLU,
    x::Vector,
)
    y = add_variables(model, x, length(x), "moai_LeakyReLU")
    y_relu, f = add_predictor(model, predictor.relu, x)
    n = predictor.negative_slope
    cons = JuMP.@constraint(model, y .== n .* x + (1 - n) .* y_relu)
    return y, Formulation(predictor, [f.variables; y], [f.constraints; cons])
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReducedSpace{<:LeakyReLU},
    x::Vector,
)
    inner_predictor = predictor.predictor
    y_relu, f_relu = add_predictor(model, ReducedSpace(inner_predictor.relu), x)
    n = inner_predictor.negative_slope
    y = JuMP.@expression(model, n .* x + (1 - n) .* y_relu)
    return y, Formulation(predictor, f_relu.variables, f_relu.constraints)
end
