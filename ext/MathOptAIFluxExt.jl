# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAIFluxExt

import Flux
import JuMP
import MathOptAI

"""
    MathOptAI.add_predictor(
        model::JuMP.AbstractModel,
        predictor::Flux.Chain,
        x::Vector;
        config::Dict = Dict{Any,Any}(),
        reduced_space::Bool = false,
    )

Add a trained neural network from Flux.jl to `model`.

## Supported layers

 * `Flux.Dense`
 * `Flux.Scale`
 * `Flux.softmax`

## Supported activation functions

 * `Flux.relu`
 * `Flux.sigmoid`
 * `Flux.softplus`
 * `Flux.tanh`

## Keyword arguments

 * `config`: a dictionary that maps supported `Flux` activation functions to
   [`AbstractPredictor`](@ref)s that control how the activation functions are
   reformulated. For example, `Flux.sigmoid => MathOptAI.Sigmoid()` or
   `Flux.relu => MathOptAI.QuadraticReLU()`.

## Example

```jldoctest
julia> using JuMP, Flux, MathOptAI

julia> chain = Flux.Chain(Flux.Dense(1 => 16, Flux.relu), Flux.Dense(16 => 1));

julia> model = Model();

julia> @variable(model, x[1:1]);

julia> y = MathOptAI.add_predictor(
           model,
           chain,
           x;
           config = Dict(Flux.relu => MathOptAI.ReLU()),
       )
1-element Vector{VariableRef}:
 moai_Affine[1]
```
"""
function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
    predictor::Flux.Chain,
    x::Vector;
    config::Dict = Dict{Any,Any}(),
    reduced_space::Bool = false,
)
    inner_predictor = MathOptAI.build_predictor(predictor; config)
    if reduced_space
        inner_predictor = MathOptAI.ReducedSpace(inner_predictor)
    end
    return MathOptAI.add_predictor(model, inner_predictor, x)
end

"""
    MathOptAI.build_predictor(
        predictor::Flux.Chain;
        config::Dict = Dict{Any,Any}(),
    )

Convert a trained neural network from Flux.jl to a [`Pipeline`](@ref).

## Supported layers

 * `Flux.Dense`
 * `Flux.Scale`
 * `Flux.softmax`

## Supported activation functions

 * `Flux.relu`
 * `Flux.sigmoid`
 * `Flux.softplus`
 * `Flux.tanh`

## Keyword arguments

 * `config`: a dictionary that maps supported `Flux` activation functions to
   [`AbstractPredictor`](@ref)s that control how the activation functions are
   reformulated. For example, `Flux.sigmoid => MathOptAI.Sigmoid()` or
   `Flux.relu => MathOptAI.QuadraticReLU()`.

## Example

```jldoctest
julia> using Flux, MathOptAI

julia> chain = Flux.Chain(Flux.Dense(1 => 16, Flux.relu), Flux.Dense(16 => 1));

julia> MathOptAI.build_predictor(
           chain;
           config = Dict(Flux.relu => MathOptAI.ReLU()),
       )
Pipeline with layers:
 * Affine(A, b) [input: 1, output: 16]
 * ReLU()
 * Affine(A, b) [input: 16, output: 1]

julia> MathOptAI.build_predictor(
           chain;
           config = Dict(Flux.relu => MathOptAI.ReLUQuadratic()),
       )
Pipeline with layers:
 * Affine(A, b) [input: 1, output: 16]
 * ReLUQuadratic()
 * Affine(A, b) [input: 16, output: 1]
```
"""
function MathOptAI.build_predictor(
    predictor::Flux.Chain;
    config::Dict = Dict{Any,Any}(),
)
    inner_predictor = MathOptAI.Pipeline(MathOptAI.AbstractPredictor[])
    for layer in predictor.layers
        _add_predictor(inner_predictor, layer, config)
    end
    return inner_predictor
end

_default(::typeof(identity)) = nothing
_default(::Any) = missing
_default(::typeof(Flux.relu)) = MathOptAI.ReLU()
_default(::typeof(Flux.sigmoid)) = MathOptAI.Sigmoid()
_default(::typeof(Flux.softplus)) = MathOptAI.SoftPlus()
_default(::typeof(Flux.softmax)) = MathOptAI.SoftMax()
_default(::typeof(Flux.tanh)) = MathOptAI.Tanh()

function _add_predictor(
    predictor::MathOptAI.Pipeline,
    activation::Function,
    config::Dict,
)
    layer = get(config, activation, _default(activation))
    if layer === nothing
        # Do nothing: a linear activation
    elseif layer === missing
        error("Unsupported activation function: $activation")
    else
        push!(predictor.layers, layer)
    end
    return
end

function _add_predictor(
    predictor::MathOptAI.Pipeline,
    layer::Flux.Dense,
    config::Dict,
)
    push!(predictor.layers, MathOptAI.Affine(layer.weight, layer.bias))
    _add_predictor(predictor, layer.σ, config)
    return
end

function _add_predictor(
    predictor::MathOptAI.Pipeline,
    layer::Flux.Scale,
    config::Dict,
)
    push!(predictor.layers, MathOptAI.Scale(layer.scale, layer.bias))
    _add_predictor(predictor, layer.σ, config)
    return
end

end  # module
