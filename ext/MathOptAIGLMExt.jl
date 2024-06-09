# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module MathOptAIGLMExt

import GLM
import JuMP
import MathOptAI

"""
    MathOptAI.add_predictor(
        model::JuMP.Model,
        predictor::GLM.LinearModel,
        x::Vector,
    )

Add a trained linear model from GLM.jl to `model`.

## Example

```jldoctest
julia> using GLM, JuMP, MathOptAI

julia> X, Y = rand(10, 2), rand(10);

julia> model_glm = GLM.lm(X, Y);

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> y = MathOptAI.add_predictor(model, model_glm, x)
1-element Vector{VariableRef}:
 omelette_Affine[1]
```
"""
function MathOptAI.add_predictor(
    model::JuMP.Model,
    predictor::GLM.LinearModel,
    x::Vector,
)
    inner_predictor = MathOptAI.Affine(GLM.coef(predictor))
    return MathOptAI.add_predictor(model, inner_predictor, x)
end

"""
    MathOptAI.add_predictor(
        model::JuMP.Model,
        predictorr::GLM.GeneralizedLinearModel{
            GLM.GlmResp{Vector{Float64},GLM.Bernoulli{Float64},GLM.LogitLink},
        },
        x::Vector,
    )

Add a trained logistic regression model from GLM.jl to `model`.

## Example

```jldoctest
julia> using GLM, JuMP, MathOptAI

julia> X, Y = rand(10, 2), rand(Bool, 10);

julia> model_glm = GLM.glm(X, Y, GLM.Bernoulli());

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> y = MathOptAI.add_predictor(model, model_glm, x)
1-element Vector{VariableRef}:
 omelette_Sigmoid[1]
```
"""
function MathOptAI.add_predictor(
    model::JuMP.Model,
    predictor::GLM.GeneralizedLinearModel{
        GLM.GlmResp{Vector{Float64},GLM.Bernoulli{Float64},GLM.LogitLink},
    },
    x::Vector,
)
    inner_predictor = MathOptAI.Pipeline(
        MathOptAI.Affine(GLM.coef(predictor)),
        MathOptAI.Sigmoid(),
    )
    return MathOptAI.add_predictor(model, inner_predictor, x)
end

end  # module
