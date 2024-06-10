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
        predictor::Lux.Experimental.TrainState,
        x::Vector,
        config::MathOptAI.AbstractConfig = MathOptAI.DefaultConfig(),
    )

Add a trained neural network from Lux.jl to `model`.

## Supported layers

 * `Lux.Dense`

## Supported activation functions

 * `Lux.relu`
 * `Lux.sigmoid`
 * `Lux.softplus`
 * `Lux.tanh`

## Example

```jldoctest; filter=r"[┌|└].+"
julia> using JuMP, Lux, MathOptAI, Random, Optimisers

julia> predictor = Lux.Experimental.TrainState(
           Random.MersenneTwister(),
           Lux.Chain(Lux.Dense(1 => 16, Lux.relu), Lux.Dense(16 => 1)),
           Optimisers.Adam(0.03f0),
       );

julia> model = Model();

julia> @variable(model, x[1:1]);

julia> y = MathOptAI.add_predictor(model, predictor, x)
1-element Vector{VariableRef}:
 moai_Affine[1]
```
"""
function MathOptAI.add_predictor(
    model::JuMP.Model,
    predictor::Lux.Experimental.TrainState,
    x::Vector,
    config::MathOptAI.AbstractConfig = MathOptAI.DefaultConfig(),
)
    inner_predictor = MathOptAI.Pipeline(MathOptAI.AbstractPredictor[])
    for (layer, parameter) in zip(predictor.model.layers, predictor.parameters)
        _add_predictor(inner_predictor, layer, parameter, config)
    end
    return MathOptAI.add_predictor(model, inner_predictor, x)
end

_default(::typeof(identity)) = nothing
_default(::Any) = missing
_default(::typeof(Lux.relu)) = MathOptAI.ReLU()
_default(::typeof(Lux.sigmoid_fast)) = MathOptAI.Sigmoid()
_default(::typeof(Lux.softplus)) = MathOptAI.SoftPlus()
_default(::typeof(Lux.tanh_fast)) = MathOptAI.Tanh()

function _add_predictor(
    predictor::MathOptAI.Pipeline,
    activation,
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
    layer::Lux.Dense,
    p,
    config::MathOptAI.AbstractConfig,
)
    affine = MathOptAI.Affine(p.weight, vec(p.bias))
    push!(predictor.layers, MathOptAI.convert_predictor(config, affine))
    _add_predictor(predictor, layer.activation, config)
    return
end

end  # module
