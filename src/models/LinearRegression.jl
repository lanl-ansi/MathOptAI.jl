# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

struct LinearRegression <: AbstractModel
    parameters::Matrix{Float64}
end

function LinearRegression(parameters::Vector{Float64})
    return LinearRegression(reshape(parameters, 1, length(parameters)))
end

Base.size(f::LinearRegression) = size(f.parameters)

function _add_model_inner(
    opt_model::JuMP.Model,
    ml_model::LinearRegression,
    x::Vector{JuMP.VariableRef},
    y::Vector{JuMP.VariableRef},
)
    JuMP.@constraint(opt_model, ml_model.parameters * x .== y)
    return
end
