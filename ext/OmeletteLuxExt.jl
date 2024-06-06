# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module OmeletteLuxExt

import Omelette
import Lux

function _add_predictor(predictor::Omelette.Pipeline, layer::Lux.Dense, p)
    push!(predictor.layers, Omelette.LinearRegression(p.weight, vec(p.bias)))
    if layer.activation === identity
        # Do nothing
    elseif layer.activation === Lux.NNlib.relu
        push!(predictor.layers, Omelette.ReLUBigM(1e6))
    else
        error("Unsupported activation function: $x")
    end
    return
end

function Omelette.Pipeline(x::Lux.Experimental.TrainState)
    predictor = Omelette.Pipeline(Omelette.AbstractPredictor[])
    for (layer, parameter) in zip(x.model.layers, x.parameters)
        _add_predictor(predictor, layer, parameter)
    end
    return predictor
end

end #module
