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
    x::Vector;
    kwargs...,
)
    inner_predictor = MathOptAI.Affine(GLM.coef(predictor))
    return MathOptAI.add_predictor(model, inner_predictor, x; kwargs...)
end

"""
    MathOptAI.add_predictor(
        model::JuMP.Model,
        predictor::GLM.GeneralizedLinearModel{
            GLM.GlmResp{Vector{Float64},GLM.Bernoulli{Float64},GLM.LogitLink},
        },
        x::Vector;
        sigmoid::AbstractPredictor = MathOptAI.Sigmoid(),
    )

Add a trained logistic regression model from GLM.jl to `model`.

## Keyword arguments

 * `sigmoid`: the predictor to use for the sigmoid layer.

## Example

```jldoctest
julia> using GLM, JuMP, MathOptAI

julia> X, Y = rand(10, 2), rand(Bool, 10);

julia> model_glm = GLM.glm(X, Y, GLM.Bernoulli());

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> y = MathOptAI.add_predictor(
           model,
           model_glm,
           x;
           sigmoid = MathOptAI.Sigmoid(),
       )
1-element Vector{VariableRef}:
 moai_Sigmoid[1]
```
"""
function MathOptAI.add_predictor(
    model::JuMP.Model,
    predictor::GLM.GeneralizedLinearModel{
        GLM.GlmResp{Vector{Float64},GLM.Bernoulli{Float64},GLM.LogitLink},
    },
    x::Vector;
    sigmoid::MathOptAI.AbstractPredictor = MathOptAI.Sigmoid(),
    kwargs...,
)
    affine = MathOptAI.Affine(GLM.coef(predictor))
    inner_predictor = MathOptAI.Pipeline(affine, sigmoid)
    return MathOptAI.add_predictor(model, inner_predictor, x; kwargs...)
end

end  # module
