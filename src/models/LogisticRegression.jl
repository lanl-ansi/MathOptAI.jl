# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

struct LogisticRegression <: AbstractPredictor
    parameters::Matrix{Float64}
end

function LogisticRegression(parameters::Vector{Float64})
    return LogisticRegression(reshape(parameters, 1, length(parameters)))
end

Base.size(f::LogisticRegression) = size(f.parameters)

function _add_predictor_inner(
    model::JuMP.Model,
    predictor::LogisticRegression,
    x::Vector{JuMP.VariableRef},
    y::Vector{JuMP.VariableRef},
)
    JuMP.@constraint(model, 1 ./ (1 .+ exp.(-predictor.parameters * x)) .== y)
    return
end
