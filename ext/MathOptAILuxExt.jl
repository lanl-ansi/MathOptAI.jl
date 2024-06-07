# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module MathOptAILuxExt

import JuMP
import Lux
import MathOptAI

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

"""
    MathOptAI.add_predictor(
        model::JuMP.Model,
        predictor::Lux.Experimental.TrainState,
        x::Vector{JuMP.VariableRef};
        config::Dict{<:Function,<:MathOptAI.AbstractPredictor} = Dict(),
    )

Add a trained neural network from Lux.jl to `model`.

## Keyword arguments

 * `config`: a dictionary that maps `Lux` activation functions to an
   `AbstractPredictor` to control how the activation functions are reformulated.

## Example

```julia
y = MathOptAI.add_predictor(
    model,
    state,
    x;
    config = Dict(Lux.relu => MathOptAI.ReLUQuadratic()),
)
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

end  # module
