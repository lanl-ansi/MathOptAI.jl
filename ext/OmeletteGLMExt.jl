# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module OmeletteGLMExt

import Omelette
import GLM

function Omelette.LinearRegression(model::GLM.LinearModel)
    return Omelette.LinearRegression(GLM.coef(model))
end

function Omelette.LogisticRegression(
    model::GLM.GeneralizedLinearModel{
        GLM.GlmResp{Vector{Float64},GLM.Bernoulli{Float64},GLM.LogitLink},
    },
)
    return Omelette.LogisticRegression(GLM.coef(model))
end

end #module
