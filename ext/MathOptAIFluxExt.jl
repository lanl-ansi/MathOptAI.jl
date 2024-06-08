# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module MathOptAIFluxExt

import Flux
import JuMP
import MathOptAI

"""
    MathOptAI.add_predictor(
        model::JuMP.Model,
        predictor::Flux.Chain,
        x::Vector{JuMP.VariableRef};
        config::Dict{<:Function,<:MathOptAI.AbstractPredictor} = Dict(),
    )

Add a trained neural network from Flux.jl to `model`.

## Supported layers

 * `Flux.Dense`

## Supported activation funnctions

 * `Flux.relu`
 * `Flux.sigmoid`
 * `Flux.softplus`
 * `Flux.tanh`

## Keyword arguments

 * `config`: a dictionary that maps `Flux` activation functions to an
   `AbstractPredictor` to control how the activation functions are reformulated.

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
 omelette_LinearRegression[1]
```
"""
function MathOptAI.add_predictor(
    model::JuMP.Model,
    predictor::Flux.Chain,
    x::Vector{JuMP.VariableRef};
    config::Dict{<:Function,<:MathOptAI.AbstractPredictor} = Dict{
        Function,
        MathOptAI.AbstractPredictor,
    }(),
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
_default(::typeof(Flux.tanh)) = MathOptAI.Tanh()

function _add_predictor(
    predictor::MathOptAI.Pipeline,
    activation::Function,
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
    layer::Flux.Dense,
    config::Dict{<:Function,<:MathOptAI.AbstractPredictor},
)
    push!(
        predictor.layers,
        MathOptAI.LinearRegression(layer.weight, layer.bias),
    )
    _add_predictor(predictor, layer.σ, config)
    return
end

end  # module
