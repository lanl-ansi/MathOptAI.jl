# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    Conv2d(
        weight::Array{T,4},
        bias::Vector{T};
        input_size::Tuple{Int,Int,Int},
        stride::Tuple{Int,Int} = (1, 1),
        padding::Tuple{Int,Int} = (0, 0),
    ) where {T} <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents a mean pooling layer a
2-dimensional convolutional layer.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[h in 1:2, w in 1:3])
2×3 Matrix{VariableRef}:
 x[1,1]  x[1,2]  x[1,3]
 x[2,1]  x[2,2]  x[2,3]

julia> weight = reshape(collect(1.0:8.0), 2, 2, 1, 2);

julia> bias = [-1.0, -2.0];

julia> predictor = MathOptAI.Conv2d(weight, bias; input_size = (2, 3, 1))
Conv2d{Float64}((2, 3, 1), [1.0 3.0; 2.0 4.0;;;; 5.0 7.0; 6.0 8.0], [-1.0, -2.0], (1, 1), (0, 0))

julia> y, formulation = MathOptAI.add_predictor(model, predictor, vec(x));

julia> y
4-element Vector{VariableRef}:
 moai_Conv2d[1]
 moai_Conv2d[2]
 moai_Conv2d[3]
 moai_Conv2d[4]

julia> formulation
Conv2d{Float64}((2, 3, 1), [1.0 3.0; 2.0 4.0;;;; 5.0 7.0; 6.0 8.0], [-1.0, -2.0], (1, 1), (0, 0))
├ variables [4]
│ ├ moai_Conv2d[1]
│ ├ moai_Conv2d[2]
│ ├ moai_Conv2d[3]
│ └ moai_Conv2d[4]
└ constraints [4]
  ├ -4 x[1,1] - 3 x[2,1] - 2 x[1,2] - x[2,2] + moai_Conv2d[1] = -1
  ├ -4 x[1,2] - 3 x[2,2] - 2 x[1,3] - x[2,3] + moai_Conv2d[2] = -1
  ├ -8 x[1,1] - 7 x[2,1] - 6 x[1,2] - 5 x[2,2] + moai_Conv2d[3] = -2
  └ -8 x[1,2] - 7 x[2,2] - 6 x[1,3] - 5 x[2,3] + moai_Conv2d[4] = -2

julia> y, formulation =
           MathOptAI.add_predictor(model, predictor, vec(x); reduced_space = true);

julia> y
4-element Vector{AffExpr}:
 4 x[1,1] + 2 x[1,2] + 3 x[2,1] + x[2,2] - 1
 4 x[1,2] + 2 x[1,3] + 3 x[2,2] + x[2,3] - 1
 8 x[1,1] + 6 x[1,2] + 7 x[2,1] + 5 x[2,2] - 2
 8 x[1,2] + 6 x[1,3] + 7 x[2,2] + 5 x[2,3] - 2

julia> formulation
ReducedSpace(Conv2d{Float64}((2, 3, 1), [1.0 3.0; 2.0 4.0;;;; 5.0 7.0; 6.0 8.0], [-1.0, -2.0], (1, 1), (0, 0)))
├ variables [0]
└ constraints [0]
```
"""
struct Conv2d{T} <: AbstractPredictor
    input_size::Tuple{Int,Int,Int} # (height, width, channel) batch == 1
    weight::Array{T,4}
    bias::Vector{T}
    stride::Tuple{Int,Int}
    padding::Tuple{Int,Int}

    function Conv2d(
        weight::Array{T,4},
        bias::Vector{T};
        input_size::Tuple{Int,Int,Int},
        stride::Tuple{Int,Int} = (1, 1),
        padding::Tuple{Int,Int} = (0, 0),
    ) where {T}
        return new{T}(input_size, weight, bias, stride, padding)
    end
end

function (f::Conv2d)(model::JuMP.AbstractModel, x::Vector)
    (Hin, Win, C) = f.input_size
    kH, kW, Cin, Cout = size(f.weight)
    @assert Cin == C
    (pH, pW), (sH, sW) = f.padding, f.stride
    Hout = floor(Int, (Hin + 2 * pH - kH) / sH + 1)
    Wout = floor(Int, (Win + 2 * pW - kW) / sW + 1)
    X = PaddedArrayView(reshape(x, Hin, Win, C), f.padding)
    return JuMP.@expression(
        model,
        [h in 1:Hout, w in 1:Wout, c in 1:Cout],
        f.bias[c] + sum(
            f.weight[kH-m+1, kW-n+1, k, c] * X[sH*(h-1)+m, sW*(w-1)+n, k] for
            m in 1:kH, n in 1:kW, k in 1:Cin
        ),
    )
end

function add_predictor(model::JuMP.AbstractModel, predictor::Conv2d, x::Vector)
    Y = predictor(model, x)
    y = add_variables(model, x, length(Y), "moai_Conv2d")
    cons = JuMP.@constraint(model, [i in 1:length(Y)], y[i] == Y[i])
    return y, Formulation(predictor, y, cons)
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReducedSpace{<:Conv2d},
    x::Vector,
)
    return vec(predictor.predictor(model, x)), Formulation(predictor)
end
