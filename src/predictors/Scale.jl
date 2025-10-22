# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    Scale(
        scale::Vector{T},
        bias::Vector{T},
    ) where {T} <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents the relationship:
```math
y = Diag(scale)x + bias
```

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, 0 <= x[i in 1:2] <= i);

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
└ constraints [6]
  ├ moai_Scale[1] ≥ 4
  ├ moai_Scale[1] ≤ 6
  ├ moai_Scale[2] ≥ 5
  ├ moai_Scale[2] ≤ 11
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

    function Scale{T}(scale::Vector{T}, bias::Vector{T}) where {T}
        if length(scale) != length(bias)
            msg = "[Scale] the `scale` and `bias` vectors must have the same length. Got `$(length(scale))` and `$(length(bias))`."
            throw(DimensionMismatch(msg))
        end
        return new(scale, bias)
    end
end

function Scale(scale::AbstractVector{U}, bias::AbstractVector{V}) where {U,V}
    T = promote_type(U, V)
    return Scale{T}(convert(Vector{T}, scale), convert(Vector{T}, bias))
end

function Base.show(io::IO, ::Scale)
    return print(io, "Scale(scale, bias)")
end

function _check_dimension(predictor::Scale, x::Vector)
    m, n = length(predictor.scale), length(x)
    if m != n
        msg = "[Scale] mismatch between the length of the input `x` ($n) and the number of columns of the `Scale` predictor ($m)."
        throw(DimensionMismatch(msg))
    end
    return
end

function (predictor::Scale)(x::Vector)
    _check_dimension(predictor, x)
    return predictor.scale .* x .+ predictor.bias
end

function add_predictor(model::JuMP.AbstractModel, predictor::Scale, x::Vector)
    _check_dimension(predictor, x)
    m = length(predictor.scale)
    y = add_variables(model, x, m, "moai_Scale")
    bounds = get_variable_bounds.(x)
    cons = Any[]
    for (i, scale) in enumerate(predictor.scale)
        y_lb = y_ub = predictor.bias[i]
        lb, ub = bounds[i]
        y_ub += scale * ifelse(scale >= 0, ub, lb)
        y_lb += scale * ifelse(scale >= 0, lb, ub)
        set_variable_bounds(cons, y[i], y_lb, y_ub; optional = true)
        set_variable_start(
            y[i],
            scale * get_variable_start(x[i]) + predictor.bias[i],
        )
    end
    append!(cons, JuMP.@constraint(model, predictor(x) .== y))
    return y, Formulation(predictor, y, cons)
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReducedSpace{<:Scale},
    x::Vector,
)
    return predictor.predictor(x), Formulation(predictor)
end
