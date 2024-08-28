# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    OffsetScaling(
        offset::Vector{T},
        factor::Vector{T},
    ) where {T} <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents the affine relationship:
```math
f(x) = \\frac{x - offset}{factor}
```

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = MathOptAI.OffsetScaling([2.0, 3.0], [4.0, 5.0])
OffsetScaling(offset, factor)

julia> y = MathOptAI.add_predictor(model, f, x)
1-element Vector{VariableRef}:
 moai_OffsetScaling[1]
 moai_OffsetScaling[2]

julia> print(model)
Feasibility
Subject to
 x[1] - 4 moai_OffsetScaling[1] = 2
 x[2] - 5 moai_OffsetScaling[2] = 3

julia> y = MathOptAI.add_predictor(model, MathOptAI.ReducedSpace(f), x)
2-element Vector{AffExpr}:
 0.25 x[1] - 0.5
 0.2 x[2] - 0.6
```
"""
struct OffsetScaling{T} <: AbstractPredictor
    offset::Vector{T}
    factor::Vector{T}
end

function Base.show(io::IO, p::OffsetScaling)
    return print(io, "OffsetScaling(offset, factor)")
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::OffsetScaling,
    x::Vector,
)
    m = length(predictor.offset)
    y = JuMP.@variable(model, [1:m], base_name = "moai_OffsetScaling")
    bounds = _get_variable_bounds.(x)
    for i in 1:m
        y_lb = y_ub = -predictor.offset[i]
        factor = 1 / predictor.factor[i]
        lb, ub = bounds[i]
        y_ub += factor * ifelse(factor >= 0, ub, lb)
        y_lb += factor * ifelse(factor >= 0, lb, ub)
        _set_bounds_if_finite(y[i], y_lb, y_ub)
    end
    JuMP.@constraint(model, x .- predictor.offset .== predictor.factor .* y)
    return y
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReducedSpace{<:OffsetScaling},
    x::Vector,
)
    offset, factor = predictor.predictor.offset, predictor.predictor.factor
    return JuMP.@expression(model, (x .- offset) ./ factor)
end
