# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    Pipeline(layers::Vector{AbstractPredictor}) <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents the relationship:
```math
y = (l_1 \\circ \\ldots \\circ l_N)(x)
```
where \$l_i\$ are a list of other [`AbstractPredictor`](@ref)s.

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
 * ReLUQuadratic(nothing)

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
1-element Vector{VariableRef}:
 moai_ReLU[1]

julia> formulation
Affine(A, b) [input: 2, output: 1]
├ variables [1]
│ └ moai_Affine[1]
└ constraints [1]
  └ x[1] + 2 x[2] - moai_Affine[1] = 0
ReLUQuadratic(nothing)
├ variables [2]
│ ├ moai_ReLU[1]
│ └ moai_z[1]
└ constraints [4]
  ├ moai_ReLU[1] ≥ 0
  ├ moai_z[1] ≥ 0
  ├ moai_Affine[1] - moai_ReLU[1] + moai_z[1] = 0
  └ moai_ReLU[1]*moai_z[1] = 0
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

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::Pipeline,
    x::Vector,
)
    formulation = PipelineFormulation(predictor, Any[])
    for layer in predictor.layers
        x, inner_formulation = add_predictor(model, layer, x)
        push!(formulation.layers, inner_formulation)
    end
    return x, formulation
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReducedSpace{Pipeline},
    x::Vector,
)
    formulation = PipelineFormulation(predictor, Any[])
    for layer in predictor.predictor.layers
        x, inner_formulation = add_predictor(model, ReducedSpace(layer), x)
        push!(formulation.layers, inner_formulation)
    end
    return x, formulation
end
