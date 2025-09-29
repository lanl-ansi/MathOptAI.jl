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

Add a vector of `n` variables to `model` with the base name `base_name`.

## Extensions

This function is a hook for JuMP extensions to interact with MathOptAI.

Implement this method for subtypes of `model`, `predictor`,  and `x` as needed.

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

"""
    get_variable_bounds(x::JuMP.AbstractVariableRef)

Return a tuple corresponding to the `(lower, upper)` variable bounds of `x`.

If there is no bound, the value returned is `missing`.

## Extensions

This function is a hook for JuMP extensions to interact with MathOptAI.

Implement this method for subtypes of `x` as needed.
"""
get_variable_bounds(::Any) = missing, missing

function get_variable_bounds(x::JuMP.AbstractVariableRef)
    lb, ub = missing, missing
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
        T = JuMP.value_type(typeof(x))
        lb = coalesce(max(lb, zero(T)), zero(T))
        ub = coalesce(min(ub, one(T)), one(T))
    end
    return lb, ub
end

"""
    set_variable_bounds(
        cons::Vector{Any},
        x::JuMP.AbstractVariableRef,
        l::Any,
        u::Any;
        optional::Bool,
    )

Set the bounds on `x` to `l` and `u`, and `push!` their corresponding constraint
references to `cons`.

If `l` or `u` are `missing`, do not set the bound.

If `optional = true`, you may choose to silently skip setting the bounds because
they are not required for correctness.

The type of `l` and `u` depends on [`get_variable_bounds`](@ref).

## Extensions

This function is a hook for JuMP extensions to interact with MathOptAI.

Implement this method for subtypes of `x` as needed.
"""
function set_variable_bounds(
    ::Vector{Any},
    ::Any,
    l::Any,
    u::Any;
    optional::Bool,
)
    if (!ismissing(l) || !ismissing(u)) && !optional
        error("You must implement this method.")
    end
    return
end

function set_variable_bounds(
    cons::Vector{Any},
    x::JuMP.AbstractVariableRef,
    l::Any,
    u::Any;
    kwargs...,
) where {T<:Real}
    if !ismissing(l) && l > typemin(JuMP.value_type(typeof(x)))
        JuMP.set_lower_bound(x, l)
        push!(cons, JuMP.LowerBoundRef(x))
    end
    if !ismissing(u) && u < typemax(JuMP.value_type(typeof(x)))
        JuMP.set_upper_bound(x, u)
        push!(cons, JuMP.UpperBoundRef(x))
    end
    return
end

