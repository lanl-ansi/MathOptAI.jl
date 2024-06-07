# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module MathOptAIGLMExt

import JuMP
import MathOptAI
import GLM

function MathOptAI.add_predictor(
    model::JuMP.Model,
    predictor::GLM.LinearModel,
    x::Vector{JuMP.VariableRef},
)
    inner_predictor = MathOptAI.LinearRegression(GLM.coef(predictor))
    return MathOptAI.add_predictor(model, inner_predictor, x)
end

function MathOptAI.add_predictor(
    model::JuMP.Model,
    predictor::GLM.GeneralizedLinearModel{
        GLM.GlmResp{Vector{Float64},GLM.Bernoulli{Float64},GLM.LogitLink},
    },
    x::Vector{JuMP.VariableRef},
)
    inner_predictor = MathOptAI.Pipeline(
        MathOptAI.LinearRegression(GLM.coef(predictor)),
        MathOptAI.Sigmoid(),
    )
    return MathOptAI.add_predictor(model, inner_predictor, x)
end

end  # module
