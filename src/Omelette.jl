# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module Omelette

import JuMP

"""
    abstract type AbstractPredictor end

## Methods

All subtypes must implement:

 * `_add_predictor_inner`
 * `Base.size`
"""
abstract type AbstractPredictor end

Base.size(x::AbstractPredictor, i::Int) = size(x)[i]

"""
    add_predictor!(
        model::JuMP.Model,
        predictor::AbstractPredictor,
        x::Vector{JuMP.VariableRef},
        y::Vector{JuMP.VariableRef},
    )::Nothing

Add the constraint `predictor(x) .== y` to the optimization model `model`.
"""
function add_predictor!(
    model::JuMP.Model,
    predictor::AbstractPredictor,
    x::Vector{JuMP.VariableRef},
    y::Vector{JuMP.VariableRef},
)
    output_n, input_n = size(predictor)
    if length(x) != input_n
        msg = "Input vector x is length $(length(x)), expected $input_n"
        throw(DimensionMismatch(msg))
    elseif length(y) != output_n
        msg = "Output vector y is length $(length(y)), expected $output_n"
        throw(DimensionMismatch(msg))
    end
    _add_predictor_inner(model, predictor, x, y)
    return nothing
end

"""
    add_predictor(
        model::JuMP.Model,
        predictor::AbstractPredictor,
        x::Vector{JuMP.VariableRef},
    )::Vector{JuMP.VariableRef}

Return an expression for `predictor(x)` in terms of variables in the
optimization model `model`.
"""
function add_predictor(
    model::JuMP.Model,
    predictor::AbstractPredictor,
    x::Vector{JuMP.VariableRef},
)
    y = JuMP.@variable(model, [1:size(predictor, 1)], base_name = "omelette_y")
    add_predictor!(model, predictor, x, y)
    return y
end

for file in readdir(joinpath(@__DIR__, "models"); join = true)
    if endswith(file, ".jl")
        include(file)
    end
end

end # module Omelette
