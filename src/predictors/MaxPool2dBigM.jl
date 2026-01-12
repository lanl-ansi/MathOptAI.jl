# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    MaxPool2dBigM(
        kernel_size::Tuple{Int,Int};
        input_size::Tuple{Int,Int,Int},
        stride::Tuple{Int,Int} = (1, 1),
        padding::Tuple{Int,Int} = (0, 0),
    ) <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents a two-dimensinal max pooling
layer.

The `max` function is implemented as a big-M mixed-integer linear program.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[h in 1:2, w in 1:4])
2×4 Matrix{VariableRef}:
 x[1,1]  x[1,2]  x[1,3]  x[1,4]
 x[2,1]  x[2,2]  x[2,3]  x[2,4]

julia> predictor =
           MathOptAI.MaxPool2dBigM((2, 2); input_size = (2, 4, 1), M = 100.0)
MaxPool2dBigM((2, 4, 1), (2, 2), (2, 2), (0, 0), 100.0)

julia> y, formulation = MathOptAI.add_predictor(model, predictor, vec(x));

julia> y
2-element Vector{VariableRef}:
 moai_MaxPool2d[1]
 moai_MaxPool2d[2]

julia> formulation
MaxPool2dBigM((2, 4, 1), (2, 2), (2, 2), (0, 0), 100.0)
├ variables [2]
│ ├ moai_MaxPool2d[1]
│ └ moai_MaxPool2d[2]
└ constraints [2]
  ├ moai_MaxPool2d[1] - max(max(max(x[1,1], x[2,1]), x[1,2]), x[2,2]) = 0
  └ moai_MaxPool2d[2] - max(max(max(x[1,3], x[2,3]), x[1,4]), x[2,4]) = 0
```
"""
struct MaxPool2dBigM <: AbstractPredictor
    input_size::Tuple{Int,Int,Int} # (height, width, channel) batch == 1
    kernel_size::Tuple{Int,Int}
    stride::Tuple{Int,Int}
    padding::Tuple{Int,Int}
    M::Float64

    function MaxPool2dBigM(
        kernel_size::Tuple{Int,Int};
        M::Float64,
        input_size::Tuple{Int,Int,Int},
        stride::Tuple{Int,Int} = kernel_size,
        padding::Tuple{Int,Int} = (0, 0),
    )
        return new(input_size, kernel_size, stride, padding, M)
    end
end

function output_size(f::MaxPool2dBigM, input_size::NTuple{3,Int})
    (Hin, Win, C) = f.input_size
    (kH, kW), (pH, pW), (sH, sW) = f.kernel_size, f.padding, f.stride
    Hout = floor(Int, (Hin + 2 * pH - kH) / sH + 1)
    Wout = floor(Int, (Win + 2 * pW - kW) / sW + 1)
    return (Hout, Wout, C)
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::MaxPool2dBigM,
    x::Vector,
)
    (Hin, Win, C) = predictor.input_size
    (kH, kW) = predictor.kernel_size
    (pH, pW) = predictor.padding
    (sH, sW) = predictor.stride
    M = predictor.M
    Hout = floor(Int, (Hin + 2 * pH - kH) / sH + 1)
    Wout = floor(Int, (Win + 2 * pW - kW) / sW + 1)
    n = Hout * Wout * C
    X = PaddedArrayView(reshape(x, Hin, Win, C), predictor.padding)
    y = add_variables(model, x, n, "moai_MaxPool2d")
    variables, cons = copy(y), Any[]
    k = 0
    for c in 1:C, w in 1:Wout, h in 1:Hout
        k += 1
        z = add_variables(model, x, kH * kW, "moai_z[h=$h,w=$w,c=$c]")
        append!(variables, z)
        JuMP.set_binary.(z)
        for m in 1:kH, n in 1:kW
            x_mn = X[sH*(h-1)+m, sW*(w-1)+n, c]
            push!(cons, JuMP.@constraint(model, y[k] >= x_mn))
            push!(cons, JuMP.@constraint(model, y[k] <= x_mn + M * z[m+kH*(n - 1)]))
        end
        push!(cons, JuMP.@constraint(model, sum(z) == length(z) - 1))
    end
    return y, Formulation(predictor, variables, cons)
end
