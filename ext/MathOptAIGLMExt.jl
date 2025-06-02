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
julia> using JuMP, MathOptAI, GLM

julia> X, Y = rand(10, 2), rand(10);

julia> predictor = GLM.lm(X, Y);

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> y, _ = MathOptAI.add_predictor(model, predictor, x);

julia> y
1-element Vector{VariableRef}:
 moai_Affine[1]

julia> MathOptAI.build_predictor(predictor)
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
julia> using JuMP, MathOptAI, GLM

julia> X, Y = rand(10, 2), rand(Bool, 10);

julia> predictor = GLM.glm(X, Y, GLM.Bernoulli());

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> y, _ = MathOptAI.add_predictor(
           model,
           predictor,
           x;
           sigmoid = MathOptAI.Sigmoid(),
       );

julia> y
1-element Vector{VariableRef}:
 moai_Sigmoid[1]

julia> MathOptAI.build_predictor(predictor)
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

end  # module MathOptAIGLMExt
