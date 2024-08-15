# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    Pipeline(layers::Vector{AbstractPredictor}) <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents a pipeline (composition) of
nested layers:
```math
f(x) = (l_1 \\cdots l_N)(x)
```

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = MathOptAI.Pipeline(
           MathOptAI.Affine([1.0 2.0], [0.0]),
           MathOptAI.ReLUQuadratic(),
       )
Pipeline with layers:
 * Affine(A, b) [input: 2, output: 1]
 * ReLUQuadratic()

julia> y = MathOptAI.add_predictor(model, f, x)
1-element Vector{VariableRef}:
 moai_ReLU[1]

julia> print(model)
Feasibility
Subject to
 x[1] + 2 x[2] - moai_Affine[1] = 0
 moai_Affine[1] - moai_ReLU[1] + _z[1] = 0
 moai_ReLU[1]*_z[1] = 0
 moai_ReLU[1] ≥ 0
 _z[1] ≥ 0
```
"""
struct Pipeline <: AbstractPredictor
    layers::Vector{AbstractPredictor}
end

Pipeline(args::AbstractPredictor...) = Pipeline(collect(args))

function Base.show(io::IO, p::Pipeline)
    print(io, "Pipeline with layers:")
    for l in p.layers
        print(io, "\n * ")
        show(io, l)
    end
    return
end

function add_predictor(model::JuMP.Model, predictor::Pipeline, x::Vector)
    for layer in predictor.layers
        x = add_predictor(model, layer, x)
    end
    return x
end

function add_predictor(
    model::JuMP.Model,
    predictor::ReducedSpace{Pipeline},
    x::Vector,
)
    for layer in predictor.predictor.layers
        x = add_predictor(model, ReducedSpace(layer), x)
    end
    return x
end
