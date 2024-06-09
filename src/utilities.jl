# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

function _get_variable_bounds(x::JuMP.VariableRef)
    lb, ub = -Inf, Inf
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
        lb, ub = max(0.0, lb), min(1.0, ub)
    end
    return lb, ub
end

function _set_bounds_if_finite(x::JuMP.VariableRef, l::Float64, u::Float64)
    if isfinite(l)
        JuMP.set_lower_bound(x, l)
    end
    if isfinite(u)
        JuMP.set_upper_bound(x, u)
    end
    return
end

# Default fallback: provide no detail on the bounds
_get_variable_bounds(::Any) = -Inf, Inf

# Default fallback: skip setting variable bound
_set_bounds_if_finite(::Any, ::Float64, ::Float64) = nothing
