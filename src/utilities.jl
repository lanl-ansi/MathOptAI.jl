# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    add_variables(
        model::JuMP.AbstractModel,
        predictor::AbstractPredictor,
        x::Vector;
        base_name::String,
    )

!!! note
    This method is for JuMP extensions. It should not be called in regular usage
    of MathOptAI.
"""
function add_variables(
    model::JuMP.AbstractModel,
    predictor::AbstractPredictor,
    x::Vector,
    n::Int;
    base_name::String,
)
    return JuMP.@variable(model, [1:n], base_name = base_name)
end

"""
    get_bounds(x::JuMP.AbstractVariable)::Tuple

Return a tuple of the `(lower, upper)` bounds associated with variable `x`.

!!! note
    This method is for JuMP extensions. It should not be called in regular usage
    of MathOptAI.
"""
get_bounds(::Any) = -Inf, Inf

function get_bounds(x::JuMP.GenericVariableRef{T}) where {T}
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

"""
    set_bounds(x::JuMP.AbstractVariable, lower, upper)::Nothing

Set the bounds of `x` to `lower` and `upper` respectively.

!!! note
    This method is for JuMP extensions. It should not be called in regular usage
    of MathOptAI.
"""
set_bounds(::Any, ::Any, ::Any) = nothing

function set_bounds(
    x::JuMP.GenericVariableRef{T},
    l::Union{Nothing,Real},
    u::Union{Nothing,Real},
) where {T}
    if l !== nothing && l > typemin(T)
        JuMP.set_lower_bound(x, l)
    end
    if u !== nothing && u < typemax(T)
        JuMP.set_upper_bound(x, u)
    end
    return
end
