# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module Omelette

import JuMP

"""
    abstract type AbstractModel end

## Methods

All subtypes must implement:

 * `add_model_internal`
 * `Base.size`
"""
abstract type AbstractModel end

"""
    add_model(
        opt_model::JuMP.Model,
        ml_model::AbstractModel,
        x::Vector{JuMP.VariableRef},
        y::Vector{JuMP.VariableRef},
    )

Add the constraint `ml_model(x) == y` to the optimization model `opt_model`.

## Input

## Output

 * `::Nothing`

## Examples

TODO
"""
function add_model(
    opt_model::JuMP.Model,
    ml_model::AbstractModel,
    x::Vector{JuMP.VariableRef},
    y::Vector{JuMP.VariableRef},
)
    output_n, input_n = size(ml_model)
    if length(x) != input_n
        msg = "Input vector x is length $(length(x)), expected $input_n"
        throw(DimensionMismatch(msg))
    elseif length(y) != output_n
        msg = "Output vector y is length $(length(y)), expected $output_n"
        throw(DimensionMismatch(msg))
    end
    _add_model_inner(opt_model, ml_model, x, y)
    return
end

for file in readdir(joinpath(@__DIR__, "models"); join = true)
    if endswith(file, ".jl")
        include(file)
    end
end

end # module Omelette
