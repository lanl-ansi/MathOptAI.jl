# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    add_variables(
        model::JuMP.AbstractModel,
        x::Vector,
        n::Int,
        base_name::String,
    )::Vector

Add a vector of `n` variables to `model` with the base name `base_name`.

## Extensions

This function is a hook for JuMP extensions to interact with MathOptAI.

Implement this method for subtypes of `model`  and `x` as needed.

The default method is:
```julia
function add_variables(
    model::JuMP.AbstractModel,
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
)
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

"""
    get_variable_start(x::JuMP.AbstractVariableRef)

Get the primal starting value of `x`, or return `missing` if one is not set.

The return value of this function is propogated through the various
[`AbstractPredictor`](@ref)s, and the primal start of new output variables is
set using [`set_variable_start`](@ref).

## Extensions

This function is a hook for JuMP extensions to interact with MathOptAI.

Implement this method for subtypes of `x` as needed.
"""
get_variable_start(::Any) = missing

function get_variable_start(x::JuMP.GenericVariableRef)
    return something(JuMP.start_value(x), missing)
end

function get_variable_start(x::JuMP.AbstractJuMPScalar)
    return JuMP.value(get_variable_start, x)
end

get_variable_start(x::Real) = x

"""
    set_variable_start(x::JuMP.AbstractVariableRef, start::Any)

Set the primal starting value of `x` to `start`, or do nothing if `start` is
`missing`.

The input value `start` of this function is computed by propogating the primal
start of the input variables (obtained with [`get_variable_start`](@ref))
through the various [`AbstractPredictor`](@ref)s.

## Extensions

This function is a hook for JuMP extensions to interact with MathOptAI.

Implement this method for subtypes of `x` and `start` as needed.
"""
set_variable_start(::Any, ::Missing) = nothing

function set_variable_start(x::Any, start::Any)
    JuMP.set_start_value(x, start)
    return
end

function set_variable_start(
    predictor::Union{Affine,Scale,SoftMax},
    x::Vector,
    y::Vector,
)
    set_variable_start.(y, predictor(get_variable_start.(x)))
    return
end

function set_variable_start(
    predictor::Union{GELU,ReLU,Sigmoid,SoftPlus,Tanh},
    x::Vector,
    y::Vector,
)
    set_variable_start.(y, predictor.(get_variable_start.(x)))
    return
end
