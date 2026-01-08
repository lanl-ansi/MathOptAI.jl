# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

struct Conv2d{T} <: AbstractPredictor
    input_size::Tuple{Int,Int,Int} # (height, width, channel) batch == 1
    weight::Array{T,4}
    bias::Vector{T}
    pad::NTuple{4,Int}
    stride::Tuple{Int,Int}
end

function add_predictor(model::JuMP.AbstractModel, predictor::Conv2d, x::Vector)
    return y, Formulation(predictor, y, cons)
end
