# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    Scale(
        scale::Vector{T},
        bias::Vector{T},
    ) where {T} <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents the affine relationship:
```math
f(x) = scale .* x .+ bias
```

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = MathOptAI.Scale([2.0, 3.0], [4.0, 5.0])
Scale(scale, bias)

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
2-element Vector{VariableRef}:
 moai_Scale[1]
 moai_Scale[2]

julia> formulation
Scale(scale, bias)
├ variables [2]
│ ├ moai_Scale[1]
│ └ moai_Scale[2]
└ constraints [2]
  ├ 2 x[1] - moai_Scale[1] = -4
  └ 3 x[2] - moai_Scale[2] = -5

julia> y, formulation =
           MathOptAI.add_predictor(model, MathOptAI.ReducedSpace(f), x);

julia> y
2-element Vector{AffExpr}:
 2 x[1] + 4
 3 x[2] + 5

julia> formulation
ReducedSpace(Scale(scale, bias))
├ variables [0]
└ constraints [0]
```
"""
struct Scale{T} <: AbstractPredictor
    scale::Vector{T}
    bias::Vector{T}
end

function Base.show(io::IO, ::Scale)
    return print(io, "Scale(scale, bias)")
end

function add_predictor(model::JuMP.AbstractModel, predictor::Scale, x::Vector)
    m = length(predictor.scale)
    y = JuMP.@variable(model, [1:m], base_name = "moai_Scale")
    bounds = _get_variable_bounds.(x)
    for (i, scale) in enumerate(predictor.scale)
        y_lb = y_ub = predictor.bias[i]
        lb, ub = bounds[i]
        y_ub += scale * ifelse(scale >= 0, ub, lb)
        y_lb += scale * ifelse(scale >= 0, lb, ub)
        _set_bounds_if_finite(y[i], y_lb, y_ub)
    end
    cons = JuMP.@constraint(model, predictor.scale .* x .+ predictor.bias .== y)
    return y, Formulation(predictor, y, cons)
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReducedSpace{<:Scale},
    x::Vector,
)
    scale, bias = predictor.predictor.scale, predictor.predictor.bias
    y = JuMP.@expression(model, scale .* x .+ bias)
    return y, Formulation(predictor)
end
