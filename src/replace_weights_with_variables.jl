# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    replace_weights_with_variables(
        model::JuMP.AbstractModel,
        predictor::AbstractPredictor,
    )

Convert `predictor` with trained weights into a predictor in which the weights
are JuMP decision variables.

This function is useful when you wish to use constrained optimization to train
small to moderate neural networks.

## Example

```jldoctest
julia> using JuMP, Flux, MathOptAI

julia> chain = Flux.Chain(Flux.Dense(2 => 3), Flux.softmax);

julia> model = Model();

julia> @variable(model, x[i in 1:2] == i);

julia> predictor = MathOptAI.build_predictor(chain)
Pipeline with layers:
 * Affine(A, b) [input: 2, output: 3]
 * SoftMax()

julia> predictor = MathOptAI.replace_weights_with_variables(model, predictor)
Pipeline with layers:
 * Affine(A, b) [input: 2, output: 3]
 * SoftMax()

julia> y, _ = MathOptAI.add_predictor(model, predictor, x);
```
"""
function replace_weights_with_variables(
    ::JuMP.AbstractModel,
    predictor::AbstractPredictor,
)
    return predictor
end

# Affine

function replace_weights_with_variables(
    model::JuMP.AbstractModel,
    predictor::Affine,
)
    m, n = size(predictor.A)
    A = JuMP.@variable(model, [i in 1:m, j in 1:n], start = predictor.A[i, j])
    b = JuMP.@variable(model, [i in 1:m], start = predictor.b[i])
    return Affine(A, b)
end

# Pipeline

function replace_weights_with_variables(
    model::JuMP.AbstractModel,
    predictor::Pipeline,
)
    return Pipeline(replace_weights_with_variables.(model, predictor.layers))
end

# Scale

function replace_weights_with_variables(
    model::JuMP.AbstractModel,
    predictor::Scale,
)
    m = length(predictor.scale)
    scale = JuMP.@variable(model, [i in 1:m], start = predictor.scale[i])
    bias = JuMP.@variable(model, [i in 1:m], start = predictor.bias[i])
    return Scale(scale, bias)
end
