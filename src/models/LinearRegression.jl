# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

struct LinearRegression <: AbstractPredictor
    parameters::Matrix{Float64}
end

function LinearRegression(parameters::Vector{Float64})
    return LinearRegression(reshape(parameters, 1, length(parameters)))
end

Base.size(f::LinearRegression) = size(f.parameters)

function _add_predictor_inner(
    model::JuMP.Model,
    predictor::LinearRegression,
    x::Vector{JuMP.VariableRef},
    y::Vector{JuMP.VariableRef},
)
    JuMP.@constraint(model, predictor.parameters * x .== y)
    return
end
