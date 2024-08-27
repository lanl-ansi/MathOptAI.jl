# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

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
    x::JuMP.GenericVariableRef{T},
    l::Union{Nothing,T},
    u::Union{Nothing,T},
) where {T}
    if l !== nothing && l > typemin(T)
        JuMP.set_lower_bound(x, l)
    end
    if u !== nothing && u < typemax(T)
        JuMP.set_upper_bound(x, u)
    end
    return
end

# Default fallback: provide no detail on the bounds
_get_variable_bounds(::Any) = -Inf, Inf

# Default fallback: skip setting variable bound
_set_bounds_if_finite(::Any, ::Any, ::Any) = nothing
