# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

function _get_variable_bounds(x::Vector{JuMP.VariableRef})
    lb, ub = fill(-Inf, length(x)), fill(Inf, length(x))
    for i in 1:length(x)
        if JuMP.has_upper_bound(x[i])
            ub[i] = JuMP.upper_bound(x[i])
        end
        if JuMP.has_lower_bound(x[i])
            lb[i] = JuMP.lower_bound(x[i])
        end
        if JuMP.is_fixed(x[i])
            lb[i] = ub[i] = JuMP.fix_value(x[i])
        end
        if JuMP.is_binary(x[i])
            lb[i] = max(0.0, lb[i])
            ub[i] = min(1.0, ub[i])
        end
    end
    return lb, ub
end

function _set_bounds_if_finite(x, l, u)
    if isfinite(l)
        JuMP.set_lower_bound(x, l)
    end
    if isfinite(u)
        JuMP.set_upper_bound(x, u)
    end
    return
end
