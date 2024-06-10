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
        model::JuMP.Model,
        predictor::Flux.Chain,
        x::Vector,
        config::MathOptAI.AbstractConfig = MathOptAI.DefaultConfig(),
    )

Add a trained neural network from Flux.jl to `model`.

## Supported layers

 * `Flux.Dense`
 * `Flux.softmax`

## Supported activation functions

 * `Flux.relu`
 * `Flux.sigmoid`
 * `Flux.softplus`
 * `Flux.tanh`

## Example

```jldoctest
julia> using JuMP, Flux, MathOptAI

julia> chain = Flux.Chain(Flux.Dense(1 => 16, Flux.relu), Flux.Dense(16 => 1));

julia> model = Model();

julia> @variable(model, x[1:1]);

julia> y = MathOptAI.add_predictor(model, chain, x)
1-element Vector{VariableRef}:
 moai_Affine[1]
```
"""
function MathOptAI.add_predictor(
    model::JuMP.Model,
    predictor::Flux.Chain,
    x::Vector,
    config::MathOptAI.AbstractConfig = MathOptAI.DefaultConfig(),
)
    inner_predictor = MathOptAI.Pipeline(MathOptAI.AbstractPredictor[])
    for layer in predictor.layers
        _add_predictor(inner_predictor, layer, config)
    end
    return MathOptAI.add_predictor(model, inner_predictor, x)
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
    config::MathOptAI.AbstractConfig,
)
    layer = _default(activation)
    if layer === nothing
        # Do nothing: a linear activation
    elseif layer === missing
        error("Unsupported activation function: $activation")
    else
        push!(predictor.layers, MathOptAI.convert_predictor(config, layer))
    end
    return
end

function _add_predictor(
    predictor::MathOptAI.Pipeline,
    layer::Flux.Dense,
    config::MathOptAI.AbstractConfig,
)
    affine = MathOptAI.Affine(layer.weight, layer.bias)
    push!(predictor.layers, MathOptAI.convert_predictor(config, affine))
    _add_predictor(predictor, layer.σ, config)
    return
end

end  # module
