# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    replace_weights_with_variables(
        model::JuMP.AbstractModel,
        predictor::AbstractPredictor;
        filter::Function = Returns(true),
    )

Convert `predictor` with trained weights into a predictor in which the weights
are JuMP decision variables.

This function is useful when you wish to use constrained optimization to train
small to moderate neural networks.

!!! warning
    This function is experimental and it may change in any future release. If
    you use this feature, please open a GitHub issue and let us know your
    thoughts.

## Keyword arguments

- `filter`: a function with the signature
  `filter(::MathOptAI.AbstractPredictor)::Bool` and returns `true` if we should
  replace with weights with decision variables.

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

julia> predictor = MathOptAI.replace_weights_with_variables(
           model,
           predictor;
           filter = l -> l isa MathOptAI.Affine,
       )
Pipeline with layers:
 * Affine(A, b) [input: 2, output: 3]
 * SoftMax()

julia> y, _ = MathOptAI.add_predictor(model, predictor, x);
```

Instead of using the `filter` argument, you can also modify only some layers:
```jldoctest
julia> using JuMP, Flux, MathOptAI

julia> chain = Flux.Chain(Flux.Dense(1 => 2), Flux.relu, Flux.Dense(2 => 1));

julia> model = Model();

julia> @variable(model, x[i in 1:1]);

julia> predictor = MathOptAI.build_predictor(chain)
Pipeline with layers:
 * Affine(A, b) [input: 1, output: 2]
 * ReLU()
 * Affine(A, b) [input: 2, output: 1]

julia> predictor.layers[3] =
           MathOptAI.replace_weights_with_variables(model, predictor.layers[3])
Affine(A, b) [input: 2, output: 1]
```
"""
function replace_weights_with_variables(
    ::JuMP.AbstractModel,
    predictor::AbstractPredictor;
    kwargs...,
)
    return predictor
end

# Affine

function replace_weights_with_variables(
    model::JuMP.AbstractModel,
    predictor::Affine;
    kwargs...,
)
    m, n = size(predictor.A)
    A = JuMP.@variable(model, [i in 1:m, j in 1:n], start = predictor.A[i, j])
    b = JuMP.@variable(model, [i in 1:m], start = predictor.b[i])
    return Affine(A, b)
end

# Pipeline

function replace_weights_with_variables(
    model::JuMP.AbstractModel,
    predictor::Pipeline;
    filter::Function = Returns(true),
)
    layers = AbstractPredictor[]
    for layer in predictor.layers
        if filter(layer) || layer isa Pipeline
            push!(layers, replace_weights_with_variables(model, layer; filter))
        else
            push!(layers, layer)
        end
    end
    return Pipeline(layers)
end

# Scale

function replace_weights_with_variables(
    model::JuMP.AbstractModel,
    predictor::Scale;
    kwargs...,
)
    m = length(predictor.scale)
    scale = JuMP.@variable(model, [i in 1:m], start = predictor.scale[i])
    bias = JuMP.@variable(model, [i in 1:m], start = predictor.bias[i])
    return Scale(scale, bias)
end
