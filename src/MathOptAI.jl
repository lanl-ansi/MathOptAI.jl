# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAI

import Distributions
import JuMP
import MathOptInterface as MOI

"""
    abstract type AbstractPredictor end

An abstract type representing different types of prediction models.

## Methods

All subtypes must implement:

 * [`add_predictor`](@ref)
"""
abstract type AbstractPredictor end

"""
    abstract type AbstractFormulation end

An abstract type representing different formulations.
"""
abstract type AbstractFormulation end

"""
    struct Formulation{P<:AbstractPredictor} <: AbstractFormulation
        predictor::P
        variables::Vector{Any}
        constraints::Vector{Any}
    end

## Fields

 * `predictor`: the predictor object used to build the formulation
 * `variables`: a vector of new decision variables added to the model
 * `constraints`: a vector of new constraints added to the model

Check the docstring of the predictor for an explanation of the formulation and
the order of the elements in `.variables` and `.constraints`.
"""
struct Formulation{P<:AbstractPredictor} <: AbstractFormulation
    predictor::P
    variables::Vector{Any}
    constraints::Vector{Any}
end

function Formulation(
    predictor::P,
    variables,
    constraints,
) where {P<:AbstractPredictor}
    return Formulation(
        predictor,
        convert(Vector{Any}, variables),
        convert(Vector{Any}, constraints),
    )
end

function Formulation(predictor::AbstractPredictor)
    return Formulation(predictor, Any[], Any[])
end

function Base.show(io::IO, formulation::Formulation)
    println(io, formulation.predictor)
    println(io, "├ variables [$(length(formulation.variables))]")
    for (i, v) in enumerate(formulation.variables)
        s = i == length(formulation.variables) ? "└" : "├"
        println(io, "│ $s ", v)
    end
    print(io, "└ constraints [$(length(formulation.constraints))]")
    for (i, c) in enumerate(formulation.constraints)
        s = i == length(formulation.constraints) ? "└" : "├"
        print(io, "\n  $s ", c)
    end
    return
end

"""
    struct PipelineFormulation{P<:AbstractPredictor} <: AbstractFormulation
        predictor::P
        layers::Vector{Any}
    end

## Fields

 * `predictor`: the predictor object used to build the formulation
 * `layers`: the formulation associated with each of the layers in the pipeline
"""
struct PipelineFormulation{P<:AbstractPredictor} <: AbstractFormulation
    predictor::P
    layers::Vector{Any}
end

function PipelineFormulation(predictor::P, layers) where {P<:AbstractPredictor}
    return PipelineFormulation(predictor, convert(Vector{Any}, layers))
end

function Base.show(io::IO, formulation::PipelineFormulation)
    for (i, c) in enumerate(formulation.layers)
        println(io, c)
    end
    return
end

"""
    add_predictor(
        model::JuMP.AbstractModel,
        predictor::AbstractPredictor,
        x::Vector,
    )::Vector

Return a `Vector` representing `y` such that `y = predictor(x)`.

The element type of `x` is deliberately unspecified. The vector `x` may contain
any mix of scalar constants, JuMP decision variables, and scalar JuMP functions
like `AffExpr`, `QuadExpr`, or `NonlinearExpr`.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = MathOptAI.Affine([2.0, 3.0])
Affine(A, b) [input: 2, output: 1]

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
1-element Vector{VariableRef}:
 moai_Affine[1]

julia> formulation
Affine(A, b) [input: 2, output: 1]
├ variables [1]
│ └ moai_Affine[1]
└ constraints [1]
  └ 2 x[1] + 3 x[2] - moai_Affine[1] = 0
```
"""
function add_predictor end

"""
    add_predictor(model::JuMP.AbstractModel, predictor, x::Matrix)

Return a `Matrix`, representing `y` such that `y[:, i] = predictor(x[:, i])` for
each columnn `i`.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2, 1:3]);

julia> f = MathOptAI.Affine([2.0, 3.0])
Affine(A, b) [input: 2, output: 1]

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
1×3 Matrix{VariableRef}:
 moai_Affine[1]  moai_Affine[1]  moai_Affine[1]

julia> formulation
Affine(A, b) [input: 2, output: 1]
├ variables [1]
│ └ moai_Affine[1]
└ constraints [1]
  └ 2 x[1,1] + 3 x[2,1] - moai_Affine[1] = 0
Affine(A, b) [input: 2, output: 1]
├ variables [1]
│ └ moai_Affine[1]
└ constraints [1]
  └ 2 x[1,2] + 3 x[2,2] - moai_Affine[1] = 0
Affine(A, b) [input: 2, output: 1]
├ variables [1]
│ └ moai_Affine[1]
└ constraints [1]
  └ 2 x[1,3] + 3 x[2,3] - moai_Affine[1] = 0
```
"""
function add_predictor(model::JuMP.AbstractModel, predictor, x::Matrix)
    inner_predictor = build_predictor(predictor)
    ret = map(j -> add_predictor(model, inner_predictor, x[:, j]), 1:size(x, 2))
    formulation = PipelineFormulation(inner_predictor, last.(ret))
    return reduce(hcat, first.(ret)), formulation
end

"""
    build_predictor(extension; kwargs...)::AbstractPredictor

A uniform interface to convert various extension types to an
[`AbstractPredictor`](@ref).

See the various extension docstrings for details.
"""
build_predictor(predictor::AbstractPredictor; kwargs...) = predictor

"""
    ReducedSpace(predictor::AbstractPredictor)

A wrapper type for other predictors that implement a reduced-space formulation.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> predictor = MathOptAI.ReducedSpace(MathOptAI.ReLU());

julia> y, formulation = MathOptAI.add_predictor(model, predictor, x);

julia> y
2-element Vector{NonlinearExpr}:
 max(0.0, x[1])
 max(0.0, x[2])
```
"""
struct ReducedSpace{P<:AbstractPredictor} <: AbstractPredictor
    predictor::P
end

ReducedSpace(predictor::ReducedSpace) = predictor

function Base.show(io::IO, predictor::ReducedSpace)
    return print(io, "ReducedSpace(", predictor.predictor, ")")
end

include("utilities.jl")

for file in filter(
    x -> endswith(x, ".jl"),
    readdir(joinpath(@__DIR__, "predictors"); join = true),
)
    include(file)
end

for sym in names(@__MODULE__; all = true)
    if !Base.isidentifier(sym) || sym in (:eval, :include)
        continue
    elseif startswith("$sym", "_")
        continue
    end
    @eval export $sym
end

end  # module MathOptAI
