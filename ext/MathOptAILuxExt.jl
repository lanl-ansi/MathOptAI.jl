# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAILuxExt

import JuMP
import Lux
import MathOptAI

"""
    MathOptAI.add_predictor(
        model::JuMP.Model,
        predictor::Tuple{<:Lux.Chain,<:NamedTuple,<:NamedTuple},
        x::Vector;
        config::Dict = Dict{Any,Any}(),
    )

Add a trained neural network from Lux.jl to `model`.

## Supported layers

 * `Lux.Dense`

## Supported activation functions

 * `Lux.relu`
 * `Lux.sigmoid`
 * `Lux.softplus`
 * `Lux.tanh`

## Keyword arguments

 * `config`: a dictionary that maps `Lux` activation functions to an
   [`AbstractPredictor`](@ref) to control how the activation functions are
   reformulated.

## Example

```jldoctest; filter=r"[┌|└].+"
julia> using JuMP, Lux, MathOptAI, Random

julia> rng = Random.MersenneTwister();

julia> chain = Lux.Chain(Lux.Dense(1 => 16, Lux.relu), Lux.Dense(16 => 1))
Chain(
    layer_1 = Dense(1 => 16, relu),     # 32 parameters
    layer_2 = Dense(16 => 1),           # 17 parameters
)         # Total: 49 parameters,
          #        plus 0 states.

julia> parameters, state = Lux.setup(rng, chain);

julia> predictor = (chain, parameters, state);

julia> model = Model();

julia> @variable(model, x[1:1]);

julia> y = MathOptAI.add_predictor(
           model,
           predictor,
           x;
           config = Dict(Lux.relu => MathOptAI.ReLU()),
       )
1-element Vector{VariableRef}:
 moai_Affine[1]
```
"""
function MathOptAI.add_predictor(
    model::JuMP.Model,
    predictor::Tuple{<:Lux.Chain,<:NamedTuple,<:NamedTuple},
    x::Vector;
    config::Dict = Dict{Any,Any}(),
    kwargs...,
)
    chain, parameters, _ = predictor
    inner_predictor = MathOptAI.Pipeline(MathOptAI.AbstractPredictor[])
    for (layer, parameter) in zip(chain.layers, parameters)
        _add_predictor(inner_predictor, layer, parameter, config)
    end
    return MathOptAI.add_predictor(model, inner_predictor, x; kwargs...)
end

_default(::typeof(identity)) = nothing
_default(::Any) = missing
_default(::typeof(Lux.relu)) = MathOptAI.ReLU()
_default(::typeof(Lux.sigmoid_fast)) = MathOptAI.Sigmoid()
_default(::typeof(Lux.softplus)) = MathOptAI.SoftPlus()
_default(::typeof(Lux.tanh_fast)) = MathOptAI.Tanh()

function _add_predictor(predictor::MathOptAI.Pipeline, activation, config::Dict)
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
    layer::Lux.Dense,
    p,
    config::Dict,
)
    push!(predictor.layers, MathOptAI.Affine(p.weight, vec(p.bias)))
    _add_predictor(predictor, layer.activation, config)
    return
end

end  # module
