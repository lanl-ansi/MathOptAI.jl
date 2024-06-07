# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module OmeletteLuxExt

import JuMP
import Lux
import Omelette

function _add_predictor(
    predictor::Omelette.Pipeline,
    layer::Lux.Dense,
    p;
    relu::Omelette.AbstractPredictor,
)
    push!(predictor.layers, Omelette.LinearRegression(p.weight, vec(p.bias)))
    if layer.activation === identity
        # Do nothing
    elseif layer.activation === Lux.NNlib.relu
        push!(predictor.layers, relu)
    else
        error("Unsupported activation function: $x")
    end
    return
end

"""
    Omelette.add_predictor(
        model::JuMP.Model,
        predictor::Lux.Experimental.TrainState,
        x::Vector{JuMP.VariableRef};
        relu::Omelette.AbstractPredictor = Omelette.ReLU(),
    )

Add a trained neural network from Lux.jl to `model`.

## Keyword arguments

 * `relu`: the predictor to use for ReLU layers
"""
function Omelette.add_predictor(
    model::JuMP.Model,
    predictor::Lux.Experimental.TrainState,
    x::Vector{JuMP.VariableRef};
    relu::Omelette.AbstractPredictor = Omelette.ReLU(),
)
    inner_predictor = Omelette.Pipeline(Omelette.AbstractPredictor[])
    for (layer, parameter) in zip(predictor.model.layers, predictor.parameters)
        _add_predictor(inner_predictor, layer, parameter; relu)
    end
    return Omelette.add_predictor(model, inner_predictor, x)
end

end #module
