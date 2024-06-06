# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

struct ReLU <: AbstractPredictor
    dimension::Int
    M::Float64
end

Base.size(x::ReLU) = (x.dimension, x.dimension)

function _add_predictor_inner(
    model::JuMP.Model,
    predictor::ReLU,
    x::Vector{JuMP.VariableRef},
    y::Vector{JuMP.VariableRef},
)
    # y = max(0, x)
    z = JuMP.@variable(model, [1:length(x)], Bin)
    JuMP.@constraint(model, y .>= 0)
    JuMP.@constraint(model, y .>= x)
    JuMP.@constraint(model, y .<= predictor.M * z)
    JuMP.@constraint(model, y .<= x .+ predictor.M * (1 .- z))
    return
end
