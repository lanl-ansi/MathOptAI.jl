# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    MaxPool2d(
        kernel_size::Tuple{Int,Int};
        input_size::Tuple{Int,Int,Int},
        stride::Tuple{Int,Int} = (1, 1),
        padding::Tuple{Int,Int} = (0, 0),
    ) <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents a two-dimensinal max pooling
layer.

The `max` function is implemented as a non-smooth nonlinear constraint.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[h in 1:2, w in 1:4])
2×4 Matrix{VariableRef}:
 x[1,1]  x[1,2]  x[1,3]  x[1,4]
 x[2,1]  x[2,2]  x[2,3]  x[2,4]

julia> predictor = MathOptAI.MaxPool2d((2, 2); input_size = (2, 4, 1))
MaxPool2d((2, 4, 1), (2, 2), (2, 2), (0, 0))

julia> y, formulation = MathOptAI.add_predictor(model, predictor, vec(x));

julia> y
2-element Vector{VariableRef}:
 moai_MaxPool2d[1]
 moai_MaxPool2d[2]

julia> formulation
MaxPool2d((2, 4, 1), (2, 2), (2, 2), (0, 0))
├ variables [2]
│ ├ moai_MaxPool2d[1]
│ └ moai_MaxPool2d[2]
└ constraints [2]
  ├ moai_MaxPool2d[1] - max(max(max(x[1,1], x[2,1]), x[1,2]), x[2,2]) = 0
  └ moai_MaxPool2d[2] - max(max(max(x[1,3], x[2,3]), x[1,4]), x[2,4]) = 0

julia> y, formulation =
           MathOptAI.add_predictor(model, predictor, vec(x); reduced_space = true);

julia> y
2-element Vector{NonlinearExpr}:
 max(max(max(x[1,1], x[2,1]), x[1,2]), x[2,2])
 max(max(max(x[1,3], x[2,3]), x[1,4]), x[2,4])

julia> formulation
ReducedSpace(MaxPool2d((2, 4, 1), (2, 2), (2, 2), (0, 0)))
├ variables [0]
└ constraints [0]
```
"""
struct MaxPool2d <: AbstractPredictor
    input_size::Tuple{Int,Int,Int} # (height, width, channel) batch == 1
    kernel_size::Tuple{Int,Int}
    stride::Tuple{Int,Int}
    padding::Tuple{Int,Int}

    function MaxPool2d(
        kernel_size::Tuple{Int,Int};
        input_size::Tuple{Int,Int,Int},
        stride::Tuple{Int,Int} = kernel_size,
        padding::Tuple{Int,Int} = (0, 0),
    )
        return new(input_size, kernel_size, stride, padding)
    end
end

function (f::MaxPool2d)(model::JuMP.AbstractModel, x::Vector)
    (Hin, Win, C) = f.input_size
    (kH, kW), (pH, pW), (sH, sW) = f.kernel_size, f.padding, f.stride
    Hout = floor(Int, (Hin + 2 * pH - kH) / sH + 1)
    Wout = floor(Int, (Win + 2 * pW - kW) / sW + 1)
    X = PaddedArrayView(reshape(x, Hin, Win, C), f.padding)
    return JuMP.@expression(
        model,
        [h in 1:Hout, w in 1:Wout, c in 1:C],
        maximum(X[sH*(h-1)+m, sW*(w-1)+n, c] for m in 1:kH, n in 1:kW),
    )
end

function output_size(f::MaxPool2d, input_size::NTuple{3,Int})
    (Hin, Win, C) = f.input_size
    (kH, kW), (pH, pW), (sH, sW) = f.kernel_size, f.padding, f.stride
    Hout = floor(Int, (Hin + 2 * pH - kH) / sH + 1)
    Wout = floor(Int, (Win + 2 * pW - kW) / sW + 1)
    return (Hout, Wout, C)
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::MaxPool2d,
    x::Vector,
)
    Y = predictor(model, x)
    y = add_variables(model, x, length(Y), "moai_MaxPool2d")
    cons = JuMP.@constraint(model, [i in 1:length(Y)], y[i] == Y[i])
    return y, Formulation(predictor, y, cons)
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReducedSpace{MaxPool2d},
    x::Vector,
)
    return vec(predictor.predictor(model, x)), Formulation(predictor)
end
