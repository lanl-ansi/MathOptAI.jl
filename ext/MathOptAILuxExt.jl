# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAILuxExt

import JuMP
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

 * `config`: a dictionary that maps supported `Lux` activation functions to
   [`AbstractPredictor`](@ref)s that control how the activation functions are
   reformulated. For example, `Lux.sigmoid => MathOptAI.Sigmoid()` or
   `Lux.relu => MathOptAI.QuadraticReLU()`.

## Example

```jldoctest; filter=r"[┌|└].+"
julia> using JuMP, MathOptAI, Lux, Random

julia> rng = Random.MersenneTwister();

julia> chain = Lux.Chain(Lux.Dense(1 => 16, Lux.relu), Lux.Dense(16 => 1))
Chain(
    layer_1 = Dense(1 => 16, relu),     # 32 parameters
    layer_2 = Dense(16 => 1),           # 17 parameters
)         # Total: 49 parameters,
          #        plus 0 states.

julia> parameters, state = Lux.setup(rng, chain);

julia> model = Model();

julia> @variable(model, x[1:1]);

julia> y, _ = MathOptAI.add_predictor(
           model,
           (chain, parameters, state),
           x;
           config = Dict(Lux.relu => MathOptAI.ReLU()),
       );

julia> y
1-element Vector{VariableRef}:
 moai_Affine[1]

julia> MathOptAI.build_predictor(
           (chain, parameters, state);
           config = Dict(Lux.relu => MathOptAI.ReLU()),
       )
Pipeline with layers:
 * Affine(A, b) [input: 1, output: 16]
 * ReLU()
 * Affine(A, b) [input: 16, output: 1]

julia> MathOptAI.build_predictor(
           (chain, parameters, state);
           config = Dict(Lux.relu => MathOptAI.ReLUQuadratic()),
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
    for (layer, parameter) in zip(chain.layers, parameters)
        _build_predictor(inner_predictor, layer, parameter, config)
    end
    return inner_predictor
end

function _build_predictor(::MathOptAI.Pipeline, layer::Any, ::Any, ::Dict)
    return error("Unsupported layer: $layer")
end

_default(::typeof(identity)) = nothing
_default(::Any) = missing
_default(::typeof(Lux.relu)) = MathOptAI.ReLU()
_default(::typeof(Lux.sigmoid)) = MathOptAI.Sigmoid()
_default(::typeof(Lux.sigmoid_fast)) = MathOptAI.Sigmoid()
_default(::typeof(Lux.softplus)) = MathOptAI.SoftPlus()
_default(::typeof(Lux.tanh)) = MathOptAI.Tanh()
_default(::typeof(Lux.tanh_fast)) = MathOptAI.Tanh()

function _build_predictor(
    predictor::MathOptAI.Pipeline,
    activation,
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

function _build_predictor(
    predictor::MathOptAI.Pipeline,
    layer::Lux.Dense,
    p::Any,
    config::Dict,
)
    push!(predictor.layers, MathOptAI.Affine(p.weight, vec(p.bias)))
    _build_predictor(predictor, layer.activation, config)
    return
end

function _build_predictor(
    predictor::MathOptAI.Pipeline,
    layer::Lux.Scale,
    p::Any,
    config::Dict,
)
    push!(predictor.layers, MathOptAI.Scale(p.weight, p.bias))
    _build_predictor(predictor, layer.activation, config)
    return
end

function _build_predictor(
    predictor::MathOptAI.Pipeline,
    ::Lux.WrappedFunction{typeof(Lux.softmax)},
    ::Any,
    config::Dict,
)
    push!(predictor.layers, get(config, Lux.softmax, MathOptAI.SoftMax()))
    return
end

end  # module MathOptAILuxExt
