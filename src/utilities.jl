# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    add_variables(
        model::JuMP.AbstractModel,
        predictor::AbstractPredictor,
        x::Vector,
        n::Int,
        base_name::String,
    )::Vector

This function is a hook for JuMP extensions to interact with MathOptAI.

The default method is:
```julia
function add_variables(
    model::JuMP.AbstractModel,
    predictor::AbstractPredictor,
    x::Vector,
    n::Int,
    base_name::String,
)
    return JuMP.@variable(model, [1:n], base_name = base_name)
end
```
Implement this method for subtypes of `model` or `x` as needed.
"""
function add_variables(
    model::JuMP.AbstractModel,
    predictor::AbstractPredictor,
    x::Vector,
    n::Int,
    base_name::String,
)
    return JuMP.@variable(model, [1:n], base_name = base_name)
end

function _get_variable_bounds(x::JuMP.GenericVariableRef{T}) where {T}
    lb, ub = typemin(T), typemax(T)
    if JuMP.has_upper_bound(x)
        ub = JuMP.upper_bound(x)
    end
    if JuMP.has_lower_bound(x)
        lb = JuMP.lower_bound(x)
    end
    if JuMP.is_fixed(x)
        lb = ub = JuMP.fix_value(x)
    end
    if JuMP.is_binary(x)
        lb, ub = max(zero(T), lb), min(one(T), ub)
    end
    return lb, ub
end

function _set_bounds_if_finite(
    cons::Vector,
    x::JuMP.GenericVariableRef{T},
    l::Union{Nothing,Real},
    u::Union{Nothing,Real},
) where {T}
    if l !== nothing && l > typemin(T)
        JuMP.set_lower_bound(x, l)
        push!(cons, JuMP.LowerBoundRef(x))
    end
    if u !== nothing && u < typemax(T)
        JuMP.set_upper_bound(x, u)
        push!(cons, JuMP.UpperBoundRef(x))
    end
    return
end

# Default fallback: provide no detail on the bounds
_get_variable_bounds(::Any) = -Inf, Inf

# Default fallback: skip setting variable bound
_set_bounds_if_finite(::Vector, ::Any, ::Any, ::Any) = nothing

function _set_direct_bounds(f::F, l, u, x::Vector, y::Vector) where {F}
    cons = Any[]
    for (xi, yi) in zip(x, y)
        x_l, x_u = _get_variable_bounds(xi)
        y_l = x_l === nothing ? l : f(x_l)
        y_u = x_u === nothing ? u : f(x_u)
        _set_bounds_if_finite(cons, yi, y_l, y_u)
    end
    return cons
end
