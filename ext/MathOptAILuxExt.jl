# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAILuxExt

import Lux
import MathOptAI

"""
    MathOptAI.build_predictor(
        predictor::Tuple{<:Lux.Chain,<:NamedTuple,<:NamedTuple};
        config::Dict = Dict{Any,Any}(),
    )

Convert a trained neural network from Lux.jl to a [`Pipeline`](@ref).

## Supported layers

 * `Lux.Dense`
 * `Lux.Scale`

## Supported activation functions

 * `Lux.relu`
 * `Lux.sigmoid`
 * `Lux.softplus`
 * `Lux.softmax`
 * `Lux.tanh`

## Keyword arguments

 * `config`: see the `Config` section below.

## Config

The `config` dictionary controls how layers in Flux are mapped to
[`AbstractPredictor`](@ref)s.

Supported keys and and example key-value pairs are:

 * `Lux.relu => MathOptAI.ReLU`
 * `Lux.sigmoid => MathOptAI.Sigmoid`
 * `Lux.sigmoid_fast => MathOptAI.Sigmoid`
 * `Lux.softmax => MathOptAI.SoftMax`
 * `Lux.softplus => MathOptAI.SoftPlus`
 * `Lux.tanh => MathOptAI.Tanh`
 * `Lux.tanh_fast => MathOptAI.Tanh`

## Example

```jldoctest; filter=r"[┌|└].+"
julia> using JuMP, MathOptAI, Lux, Random

julia> rng = Random.MersenneTwister();

julia> chain = Lux.Chain(Lux.Dense(1 => 16, Lux.relu), Lux.Dense(16 => 1))
Chain(
    layer_1 = Dense(1 => 16, relu),               # 32 parameters
    layer_2 = Dense(16 => 1),                     # 17 parameters
)         # Total: 49 parameters,
          #        plus 0 states.

julia> parameters, state = Lux.setup(rng, chain);

julia> model = Model();

julia> @variable(model, x[1:1]);

julia> y, _ = MathOptAI.add_predictor(
           model,
           (chain, parameters, state),
           x;
           config = Dict(Lux.relu => MathOptAI.ReLU),
       );

julia> y
1-element Vector{VariableRef}:
 moai_Affine[1]

julia> MathOptAI.build_predictor(
           (chain, parameters, state);
           config = Dict(Lux.relu => MathOptAI.ReLU),
       )
Pipeline with layers:
 * Affine(A, b) [input: 1, output: 16]
 * ReLU()
 * Affine(A, b) [input: 16, output: 1]

julia> MathOptAI.build_predictor(
           (chain, parameters, state);
           config = Dict(Lux.relu => MathOptAI.ReLUQuadratic),
       )
Pipeline with layers:
 * Affine(A, b) [input: 1, output: 16]
 * ReLUQuadratic(nothing)
 * Affine(A, b) [input: 16, output: 1]
```
"""
function MathOptAI.build_predictor(
    predictor::Tuple{<:Lux.Chain,<:NamedTuple,<:NamedTuple};
    config::Dict = Dict{Any,Any}(),
)
    chain, parameters, _ = predictor
    inner_predictor = MathOptAI.Pipeline(MathOptAI.AbstractPredictor[])
    for layer_params in zip(chain.layers, parameters)
        layer_p = MathOptAI.build_predictor(layer_params; config)
        if layer_p isa MathOptAI.Pipeline
            append!(inner_predictor.layers, layer_p.layers)
        else
            push!(inner_predictor.layers, layer_p)
        end
    end
    return inner_predictor
end

for (f, P) in (
    Lux.relu => MathOptAI.ReLU,
    Lux.sigmoid => MathOptAI.Sigmoid,
    Lux.sigmoid_fast => MathOptAI.Sigmoid,
    Lux.softplus => MathOptAI.SoftPlus,
    Lux.tanh => MathOptAI.Tanh,
    Lux.tanh_fast => MathOptAI.Tanh,
)
    @eval function MathOptAI.build_predictor(
        activation::typeof($f);
        config::Dict = Dict{Any,Any}(),
        kwargs...,
    )
        return get(config, activation, $P)()
    end
end

function MathOptAI.build_predictor(
    (layer, params)::Tuple{Lux.Dense,Any};
    kwargs...,
)
    p = MathOptAI.Affine(params.weight, vec(params.bias))
    σ = MathOptAI.build_predictor(layer.activation; kwargs...)
    return MathOptAI.Pipeline(p, σ)
end

function MathOptAI.build_predictor(
    (layer, params)::Tuple{Lux.Scale,Any};
    kwargs...,
)
    p = MathOptAI.Scale(params.weight, params.bias)
    σ = MathOptAI.build_predictor(layer.activation; kwargs...)
    return MathOptAI.Pipeline(p, σ)
end

function MathOptAI.build_predictor(
    ::Tuple{Lux.WrappedFunction{typeof(Lux.softmax)},Any};
    config::Dict = Dict{Any,Any}(),
)
    return get(config, Lux.softmax, MathOptAI.SoftMax)()
end

end  # module MathOptAILuxExt
