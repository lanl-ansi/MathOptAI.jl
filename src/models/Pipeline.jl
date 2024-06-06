# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

struct Pipeline <: AbstractPredictor
    layers::Vector{AbstractPredictor}
end

Base.size(x::Pipeline) = (size(last(x.layers), 1), size(first(x.layers), 2))

function _add_predictor_inner(
    model::JuMP.Model,
    predictor::Pipeline,
    x::Vector{JuMP.VariableRef},
    y::Vector{JuMP.VariableRef},
)
    for (i, layer) in enumerate(predictor.layers)
        if i == length(predictor.layers)
            add_predictor!(model, layer, x, y)
        else
            x = add_predictor(model, layer, x)
        end
    end
    return
end
