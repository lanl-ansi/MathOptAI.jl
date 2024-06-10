# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    Pipeline(layers::Vector{AbstractPredictor}) <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents a pipeline of nested layers
```math
f(x) = l_N(\\ldots(l_2(l_1(x))
```

## Example

```jldoctest
julia> using JuMP

julia> import MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = MathOptAI.Pipeline(
           MathOptAI.Affine([1.0 2.0], [0.0]),
           MathOptAI.ReLUQuadratic(),
       )
MathOptAI.Pipeline(MathOptAI.AbstractPredictor[MathOptAI.Affine([1.0 2.0], [0.0]), MathOptAI.ReLUQuadratic()])

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

function add_predictor(model::JuMP.Model, predictor::Pipeline, x::Vector)
    for layer in predictor.layers
        x = add_predictor(model, layer, x)
    end
    return x
end
