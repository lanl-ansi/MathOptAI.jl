# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module MathOptAILuxExt

import JuMP
import Lux
import MathOptAI

"""
    MathOptAI.add_predictor(
        model::JuMP.Model,
        predictor::Lux.Experimental.TrainState,
        x::Vector{JuMP.VariableRef};
        config::Dict{<:Function,<:MathOptAI.AbstractPredictor} = Dict(),
    )

Add a trained neural network from Lux.jl to `model`.

## Supported layers

 * `Lux.Dense`

## Supported activation funnctions

 * `Lux.relu`
 * `Lux.sigmoid`
 * `Lux.softplus`
 * `Lux.tanh`

## Keyword arguments

 * `config`: a dictionary that maps `Lux` activation functions to an
   `AbstractPredictor` to control how the activation functions are reformulated.

## Example

```jldoctest
julia> using JuMP, Lux, MathOptAI, Random, Optimisers

julia> predictor = Lux.Experimental.TrainState(
           Random.MersenneTwister(),
           Lux.Chain(Lux.Dense(1 => 16, Lux.relu), Lux.Dense(16 => 1)),
           Optimisers.Adam(0.03f0),
       );

julia> model = Model();

julia> @variable(model, x[1:1]);

julia> y = MathOptAI.add_predictor(
           model,
           predictor,
           x;
           config = Dict(Lux.relu => MathOptAI.ReLU()),
       )
1-element Vector{VariableRef}:
 omelette_LinearRegression[1]
```
"""
function MathOptAI.add_predictor(
    model::JuMP.Model,
    predictor::Lux.Experimental.TrainState,
    x::Vector{JuMP.VariableRef};
    config::Dict{<:Function,<:MathOptAI.AbstractPredictor} = Dict{
        Function,
        MathOptAI.AbstractPredictor,
    }(),
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
    config::Dict{<:Function,<:MathOptAI.AbstractPredictor},
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
    layer::Lux.Dense,
    p,
    config::Dict{<:Function,<:MathOptAI.AbstractPredictor},
)
    push!(predictor.layers, MathOptAI.LinearRegression(p.weight, vec(p.bias)))
    _add_predictor(predictor, layer.activation, config)
    return
end

end  # module