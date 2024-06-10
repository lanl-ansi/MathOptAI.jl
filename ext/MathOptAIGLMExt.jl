# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAIGLMExt

import GLM
import JuMP
import MathOptAI

"""
    MathOptAI.add_predictor(
        model::JuMP.Model,
        predictor::GLM.LinearModel,
        x::Vector,
        config::MathOptAI.AbstractConfig = MathOptAI.DefaultConfig(),
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
 moai_Affine[1]
```
"""
function MathOptAI.add_predictor(
    model::JuMP.Model,
    predictor::GLM.LinearModel,
    x::Vector,
    config::MathOptAI.AbstractConfig = MathOptAI.DefaultConfig(),
)
    affine = MathOptAI.Affine(GLM.coef(predictor))
    inner = MathOptAI.convert_predictor(config, affine)
    return MathOptAI.add_predictor(model, inner, x)
end

"""
    MathOptAI.add_predictor(
        model::JuMP.Model,
        predictorr::GLM.GeneralizedLinearModel{
            GLM.GlmResp{Vector{Float64},GLM.Bernoulli{Float64},GLM.LogitLink},
        },
        x::Vector,
        config::MathOptAI.AbstractConfig = MathOptAI.DefaultConfig(),
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
 moai_Sigmoid[1]
```
"""
function MathOptAI.add_predictor(
    model::JuMP.Model,
    predictor::GLM.GeneralizedLinearModel{
        GLM.GlmResp{Vector{Float64},GLM.Bernoulli{Float64},GLM.LogitLink},
    },
    x::Vector,
    config::MathOptAI.AbstractConfig = MathOptAI.DefaultConfig(),
)
    affine = MathOptAI.Affine(GLM.coef(predictor))
    inner_predictor = MathOptAI.Pipeline(
        MathOptAI.convert_predictor(config, affine),
        MathOptAI.convert_predictor(config, MathOptAI.Sigmoid()),
    )
    return MathOptAI.add_predictor(model, inner_predictor, x)
end

end  # module
