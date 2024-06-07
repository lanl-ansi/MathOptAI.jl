# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module OmeletteGLMExt

import JuMP
import Omelette
import GLM

function Omelette.add_predictor(
    model::JuMP.Model,
    predictor::GLM.LinearModel,
    x::Vector{JuMP.VariableRef},
)
    inner_predictor = Omelette.LinearRegression(GLM.coef(predictor))
    return Omelette.add_predictor(model, inner_predictor, x)
end

function Omelette.add_predictor(
    model::JuMP.Model,
    predictor::GLM.GeneralizedLinearModel{
        GLM.GlmResp{Vector{Float64},GLM.Bernoulli{Float64},GLM.LogitLink},
    },
    x::Vector{JuMP.VariableRef},
)
    inner_predictor = Omelette.Pipeline(
        Omelette.LinearRegression(GLM.coef(predictor)),
        Omelette.Sigmoid(),
    )
    return Omelette.add_predictor(model, inner_predictor, x)
end

end  # module
