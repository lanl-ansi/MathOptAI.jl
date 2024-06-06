# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module Omelette

import Distributions
import JuMP
import MathOptInterface as MOI

"""
    abstract type AbstractPredictor end

An abstract type representig different types of prediction models.

## Methods

All subtypes must implement:

 * `add_predictor`
"""
abstract type AbstractPredictor end

"""
    add_predictor(
        model::JuMP.Model,
        predictor::AbstractPredictor,
        x::Vector{JuMP.VariableRef},
    )::Vector{JuMP.VariableRef}

Return a `Vector{JuMP.VariableRef}` representing `y` such that
`y = predictor(x)`.

## Example

```jldoctest
julia> using JuMP, Omelette

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = Omelette.LinearRegression([2.0, 3.0])
Omelette.LinearRegression([2.0 3.0])

julia> y = Omelette.add_predictor(model, f, x)
1-element Vector{VariableRef}:
 omelette_y[1]

julia> print(model)
 Feasibility
 Subject to
  2 x[1] + 3 x[2] - omelette_y[1] = 0
```
"""
function add_predictor end

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

for file in readdir(joinpath(@__DIR__, "models"); join = true)
    if endswith(file, ".jl")
        include(file)
    end
end

end # module Omelette
