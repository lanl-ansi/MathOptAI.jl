# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    TAGConv(;
        weights::Vector{Matrix{T}},
        bias::Vector{T},
        edge_index::Vector{Pair{Int,Int}},
    )

An [`AbstractPredictor`](@ref) that represents a topology adaptive graph
convolutional network operator:
```math
Y = \\sum\\limits_{k=0}^K (D^{-1/2} A D^{-1/2})^k X W_k + b
```
where:

 * ``W_k`` is `weights[k+1]`
 * ``A`` is the adjacency matrix constructed from `edge_index`
 * ``b`` is the bias vector `bias`
 * ``D`` is the diagonal degree matrix.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:3, 1:2]);

julia> f = MathOptAI.TAGConv(;
           weights = Matrix{Float64}[[1 2]', [3 4]', [5 6]'],
           bias = [7.0],
           edge_index = [1 => 2, 2 => 1, 2 => 3, 3 => 2],
       );

julia> y, formulation = MathOptAI.add_predictor(model, f, vec(x));

julia> y
3-element Vector{VariableRef}:
 moai_TAGConv[1]
 moai_TAGConv[2]
 moai_TAGConv[3]

julia> formulation
TAGConv{Float64}([[1.0; 2.0;;], [3.0; 4.0;;], [5.0; 6.0;;]], [7.0], [0.0 0.7071067811865475 0.0; 0.7071067811865475 0.0 0.7071067811865475; 0.0 0.7071067811865475 0.0])
├ variables [3]
│ ├ moai_TAGConv[1]
│ ├ moai_TAGConv[2]
│ └ moai_TAGConv[3]
└ constraints [3]
  ├ -3.4999999999999996 x[1,1] - 2.1213203435596424 x[2,1] - 2.4999999999999996 x[3,1] - 4.999999999999999 x[1,2] - 2.82842712474619 x[2,2] - 2.999999999999999 x[3,2] + moai_TAGConv[1] = 7
  ├ -2.1213203435596424 x[1,1] - 5.999999999999999 x[2,1] - 2.1213203435596424 x[3,1] - 2.82842712474619 x[1,2] - 7.999999999999998 x[2,2] - 2.82842712474619 x[3,2] + moai_TAGConv[2] = 7
  └ -2.4999999999999996 x[1,1] - 2.1213203435596424 x[2,1] - 3.4999999999999996 x[3,1] - 2.999999999999999 x[1,2] - 2.82842712474619 x[2,2] - 4.999999999999999 x[3,2] + moai_TAGConv[3] = 7
```
"""
struct TAGConv{T} <: AbstractPredictor
    weights::Vector{Matrix{T}}
    bias::Vector{T}
    B::Matrix{T}

    function TAGConv(;
        weights::Vector{Matrix{T}},
        bias::Vector{T},
        edge_index::Vector{Pair{Int,Int}},
        n::Int = mapreduce(maximum, max, edge_index),
    ) where {T}
        A = zeros(T, n, n)
        for (i, j) in edge_index
            A[i, j] += one(T)
        end
        d = sqrt.(sum(A; dims = 2))
        for (i, j) in unique(edge_index)
            A[i, j] /= (d[i] * d[j])
        end
        return new{T}(weights, bias, A)
    end
end

output_size(f::TAGConv, input_size) = (size(f.B, 1), size(f.weights[end], 2))

function (f::TAGConv)(model::JuMP.AbstractModel, x)
    X = reshape(x, size(f.B, 2), size(f.weights[1], 1))
    return JuMP.@expression(
        model,
        sum(f.B^(k-1) * X * W for (k, W) in enumerate(f.weights)) .+ f.bias',
    )
end

function add_predictor(model::JuMP.AbstractModel, predictor::TAGConv, x::Vector)
    Y = predictor(model, x)
    y = add_variables(model, x, length(Y), "moai_TAGConv")
    cons = JuMP.@constraint(model, y .== vec(Y))
    return y, Formulation(predictor, y, cons)
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReducedSpace{TAGConv{T}},
    x::Vector,
) where {T}
    return vec(predictor.predictor(model, x)), Formulation(predictor)
end
