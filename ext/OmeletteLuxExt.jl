# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module OmeletteLuxExt

import JuMP
import Lux
import Omelette

_default(::typeof(identity)) = nothing
_default(::Any) = missing
_default(::typeof(Lux.relu)) = Omelette.ReLU()
_default(::typeof(Lux.sigmoid_fast)) = Omelette.Sigmoid()
_default(::typeof(Lux.softplus)) = Omelette.SoftPlus()
_default(::typeof(Lux.tanh_fast)) = Omelette.Tanh()

function _add_predictor(
    predictor::Omelette.Pipeline,
    activation,
    config::Dict{<:Function,<:Omelette.AbstractPredictor},
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
    predictor::Omelette.Pipeline,
    layer::Lux.Dense,
    p,
    config::Dict{<:Function,<:Omelette.AbstractPredictor},
)
    push!(predictor.layers, Omelette.LinearRegression(p.weight, vec(p.bias)))
    _add_predictor(predictor, layer.activation, config)
    return
end

"""
    Omelette.add_predictor(
        model::JuMP.Model,
        predictor::Lux.Experimental.TrainState,
        x::Vector{JuMP.VariableRef};
        config::Dict{<:Function,<:Omelette.AbstractPredictor} = Dict(),
    )

Add a trained neural network from Lux.jl to `model`.

## Keyword arguments

 * `config`: a dictionary that maps `Lux` activation functions to an
   `AbstractPredictor` to control how the activation functions are reformulated.

## Example

```julia
y = Omelette.add_predictor(
    model,
    state,
    x;
    config = Dict(Lux.relu => Omelette.ReLUQuadratic()),
)
```
"""
function Omelette.add_predictor(
    model::JuMP.Model,
    predictor::Lux.Experimental.TrainState,
    x::Vector{JuMP.VariableRef};
    config::Dict{<:Function,<:Omelette.AbstractPredictor} = Dict{
        Function,
        Omelette.AbstractPredictor,
    }(),
)
    inner_predictor = Omelette.Pipeline(Omelette.AbstractPredictor[])
    for (layer, parameter) in zip(predictor.model.layers, predictor.parameters)
        _add_predictor(inner_predictor, layer, parameter, config)
    end
    return Omelette.add_predictor(model, inner_predictor, x)
end

end  # module
