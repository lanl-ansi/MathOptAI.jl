# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    GCNConv(;
        weights::Matrix{T},
        bias::Vector{T},
        edge_index::Vector{Pair{Int,Int}},
    )

An [`AbstractPredictor`](@ref) that represents a graph convolutional network
operator:
```math
Y = D^{-1/2} A D^{-1/2} X W + b
```

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:3, 1:2]);

julia> f = MathOptAI.GCNConv(;
           weights = [1.0; 2.0;;],
           bias = [7.0],
           edge_index = [1 => 2, 2 => 1, 2 => 3, 3 => 2],
       );

julia> y, formulation = MathOptAI.add_predictor(model, f, vec(x));

julia> y
3-element Vector{VariableRef}:
 moai_GCNConv[1]
 moai_GCNConv[2]
 moai_GCNConv[3]

julia> formulation
GCNConv{Float64}([1.0; 2.0;;], [7.0], [0.4999999999999999 0.40824829046386296 0.0; 0.40824829046386296 0.33333333333333337 0.40824829046386296; 0.0 0.40824829046386296 0.4999999999999999])
├ variables [3]
│ ├ moai_GCNConv[1]
│ ├ moai_GCNConv[2]
│ └ moai_GCNConv[3]
└ constraints [3]
  ├ -0.4999999999999999 x[1,1] - 0.40824829046386296 x[2,1] - x[1,2] - 0.8164965809277259 x[2,2] + moai_GCNConv[1] = 7
  ├ -0.40824829046386296 x[1,1] - 0.33333333333333337 x[2,1] - 0.40824829046386296 x[3,1] - 0.8164965809277259 x[1,2] - 0.6666666666666667 x[2,2] - 0.8164965809277259 x[3,2] + moai_GCNConv[2] = 7
  └ -0.40824829046386296 x[2,1] - 0.4999999999999999 x[3,1] - 0.8164965809277259 x[2,2] - x[3,2] + moai_GCNConv[3] = 7
```
"""
struct GCNConv{T} <: AbstractPredictor
    weights::Matrix{T}
    bias::Vector{T}
    DAD::Matrix{T}

    function GCNConv(;
        weights::Matrix{T},
        bias::Vector{T},
        edge_index::Vector{Pair{Int,Int}},
        n::Int = mapreduce(maximum, max, edge_index),
    ) where {T}
        edge_index = unique(edge_index)
        A = zeros(T, n, n)
        for (i, j) in edge_index
            A[i, j] += one(T)
        end
        for i in 1:n
            if iszero(A[i, i])
                A[i, i] += one(T)
                push!(edge_index, i => i)
            end
        end
        d = sqrt.(sum(A; dims = 2))
        for (i, j) in edge_index
            A[i, j] /= (d[i] * d[j])
        end
        return new{T}(weights, bias, A)
    end
end

output_size(f::GCNConv, input_size) = (size(f.DAD, 1), size(f.weights, 2))

function (f::GCNConv)(model::JuMP.AbstractModel, x)
    X = reshape(x, size(f.DAD, 2), size(f.weights, 1))
    return JuMP.@expression(model, f.DAD * X * f.weights .+ f.bias')
end

function add_predictor(model::JuMP.AbstractModel, predictor::GCNConv, x::Vector)
    Y = predictor(model, x)
    y = add_variables(model, x, length(Y), "moai_GCNConv")
    cons = JuMP.@constraint(model, y .== vec(Y))
    return y, Formulation(predictor, y, cons)
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReducedSpace{GCNConv{T}},
    x::Vector,
) where {T}
    return vec(predictor.predictor(model, x)), Formulation(predictor)
end
