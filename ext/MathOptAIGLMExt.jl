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
        model::JuMP.AbstractModel,
        predictor::GLM.LinearModel,
        x::Vector;
        reduced_space::Bool = false,
    )

Add a trained linear model from GLM.jl to `model`.

## Example

```jldoctest
julia> using GLM, JuMP, MathOptAI

julia> X, Y = rand(10, 2), rand(10);

julia> model_glm = GLM.lm(X, Y);

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> y, _ = MathOptAI.add_predictor(model, model_glm, x);

julia> y
1-element Vector{VariableRef}:
 moai_Affine[1]
```
"""
function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
    predictor::GLM.LinearModel,
    x::Vector;
    reduced_space::Bool = false,
)
    inner_predictor = MathOptAI.build_predictor(predictor)
    if reduced_space
        inner_predictor = MathOptAI.ReducedSpace(inner_predictor)
    end
    return MathOptAI.add_predictor(model, inner_predictor, x)
end

"""
    MathOptAI.build_predictor(predictor::GLM.LinearModel)

Convert a trained linear model from GLM.jl to an [`Affine`](@ref) layer.

## Example

```jldoctest
julia> using GLM, MathOptAI

julia> X, Y = rand(10, 2), rand(10);

julia> model_glm = GLM.lm(X, Y);

julia> MathOptAI.build_predictor(model_glm)
Affine(A, b) [input: 2, output: 1]
```
"""
function MathOptAI.build_predictor(predictor::GLM.LinearModel)
    return MathOptAI.Affine(GLM.coef(predictor))
end

"""
    MathOptAI.add_predictor(
        model::JuMP.AbstractModel,
        predictor::GLM.GeneralizedLinearModel{
            GLM.GlmResp{Vector{Float64},GLM.Bernoulli{Float64},GLM.LogitLink},
        },
        x::Vector;
        sigmoid::AbstractPredictor = MathOptAI.Sigmoid(),
        reduced_space::Bool = false,
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

julia> y, _ = MathOptAI.add_predictor(
           model,
           model_glm,
           x;
           sigmoid = MathOptAI.Sigmoid(),
       );

julia> y
1-element Vector{VariableRef}:
 moai_Sigmoid[1]
```
"""
function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
    predictor::GLM.GeneralizedLinearModel{
        GLM.GlmResp{Vector{Float64},GLM.Bernoulli{Float64},GLM.LogitLink},
    },
    x::Vector;
    sigmoid::MathOptAI.AbstractPredictor = MathOptAI.Sigmoid(),
    reduced_space::Bool = false,
)
    inner_predictor = MathOptAI.build_predictor(predictor; sigmoid)
    if reduced_space
        inner_predictor = MathOptAI.ReducedSpace(inner_predictor)
    end
    return MathOptAI.add_predictor(model, inner_predictor, x)
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

julia> model_glm = GLM.glm(X, Y, GLM.Bernoulli());

julia> MathOptAI.build_predictor(model_glm)
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
