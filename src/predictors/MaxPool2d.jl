# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

struct MaxPool2d <: AbstractPredictor
    input_size::Tuple{Int,Int,Int} # (height, width, channel) batch == 1
    kernel_size::Tuple{Int,Int}
    stride::Tuple{Int,Int}
    padding::Tuple{Int,Int}
end

function (f::MaxPool2d)(x::Vector)
    (Hin, Win, C) = f.input_size
    (kH, kW), (pH, pW), (sH, sW) = f.kernel_size, f.padding, f.stride
    Hout = floor(Int, (Hin + 2 * pH - kH) / sH + 1)
    Wout = floor(Int, (Win + 2 * pW - kW) / sW + 1)
    X = PaddedArrayView(reshape(x, Hin, Win, C), f.padding)
    return [
        maximum(X[sH*(h-1)+m,sW*(w-1)+n,c] for m in 1:kH, n in 1:kW) for
        h in 1:Hout, w in 1:Wout, c in 1:C
    ]
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::MaxPool2d,
    x::Vector,
)
    Y = predictor(x)
    y = add_variables(model, x, length(Y), "moai_MaxPool2d")
    cons = JuMP.@constraint(model, [i in 1:length(Y)], y[i] == Y[i])
    return y, Formulation(predictor, y, cons)
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReducedSpace{MaxPool2d},
    x::Vector,
)
    return predictor(x), Formulation(predictor)
end
