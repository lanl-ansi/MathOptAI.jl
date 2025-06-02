# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAIGLMExt

import GLM
import JuMP
import MathOptAI

"""
    MathOptAI.build_predictor(predictor::GLM.LinearModel)

Convert a trained linear model from GLM.jl to an [`Affine`](@ref) layer.

## Example

```jldoctest
julia> using GLM, MathOptAI

julia> X, Y = rand(10, 2), rand(10);

julia> model = GLM.lm(X, Y);

julia> predictor = MathOptAI.build_predictor(model)
Affine(A, b) [input: 2, output: 1]
```
"""
function MathOptAI.build_predictor(predictor::GLM.LinearModel)
    return MathOptAI.Affine(GLM.coef(predictor))
end

"""
    MathOptAI.build_predictor(
        predictor::GLM.GeneralizedLinearModel{
            GLM.GlmResp{Vector{Float64},GLM.Bernoulli{Float64},GLM.LogitLink},
        };
        sigmoid::MathOptAI.AbstractPredictor = MathOptAI.Sigmoid(),
    )

Convert a trained logistic model from GLM.jl to a [`Pipeline`](@ref) layer.

## Keyword arguments

 * `sigmoid`: the predictor to use for the sigmoid layer.

## Example

```jldoctest
julia> using GLM, MathOptAI

julia> X, Y = rand(10, 2), rand(Bool, 10);

julia> model = GLM.glm(X, Y, GLM.Bernoulli());

julia> predictor = MathOptAI.build_predictor(model)
Pipeline with layers:
 * Affine(A, b) [input: 2, output: 1]
 * Sigmoid()
```
"""
function MathOptAI.build_predictor(
    predictor::GLM.GeneralizedLinearModel{
        GLM.GlmResp{Vector{Float64},GLM.Bernoulli{Float64},GLM.LogitLink},
    };
    sigmoid::MathOptAI.AbstractPredictor = MathOptAI.Sigmoid(),
)
    affine = MathOptAI.Affine(GLM.coef(predictor))
    return MathOptAI.Pipeline(affine, sigmoid)
end

end  # module
