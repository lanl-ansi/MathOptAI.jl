# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    GeLU() <: AbstractPredictor

## Example

TODO

```jldoctest
```
"""
struct GELU <: AbstractPredictor end

_gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::GELU,
    x::Vector,
)
    y = JuMP.@variable(model, [1:length(x)], base_name = "moai_GELU")
    gelu_lb = -0.17 # This is a valid lower bound on GELU

    cons = Any[]
    for (xi, yi) in zip(x, y)
        x_l, x_u = _get_variable_bounds(xi)
        if x_l === nothing
            y_l = -0.17
        elseif x_l >= 0.0
            y_l = _gelu(x_l)
        else
            y_l = -0.17
        end

        if x_u === nothing
            y_u = nothing
        elseif x_u >= 0.0
            y_u = _gelu(x_u)
        else
            y_u = 0.0
        end
        _set_bounds_if_finite(cons, yi, y_l, y_u)
    end

    append!(cons, JuMP.@constraint(model, y .== _gelu.(x)))
    return y, Formulation(predictor, y, cons)
end

function add_predictor(
    ::JuMP.AbstractModel,
    predictor::ReducedSpace{GELU},
    x::Vector,
)
    return _gelu.(x), Formulation(predictor)
end
