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
 * [`build_predictor`](@ref)

The following methods are optional, but encouraged:

 * [`output_size`](@ref)
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
        predictor::Any,
        x::Vector;
        reduced_space::Bool = false,
        kwargs...,
    )::Tuple{<:Vector,<:AbstractFormulation}

Return a `Vector` representing `y` such that `y = predictor(x)` and an
[`AbstractFormulation`](@ref) containing the variables and constraints that were
added to the model.

The element type of `x` is deliberately unspecified. The vector `x` may contain
any mix of scalar constants, JuMP decision variables, and scalar JuMP functions
like `AffExpr`, `QuadExpr`, or `NonlinearExpr`.

## Keyword arguments

 * `reduced_space`: if `true`, wrap `predictor` in [`ReducedSpace`](@ref) before
   adding to the model.

All other keyword arguments are passed to [`build_predictor`](@ref).

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

julia> y, formulation = MathOptAI.add_predictor(model, f, x; reduced_space = true);

julia> y
1-element Vector{AffExpr}:
 2 x[1] + 3 x[2]

julia> formulation
ReducedSpace(Affine(A, b) [input: 2, output: 1])
├ variables [0]
└ constraints [0]
```
"""
function add_predictor(
    model::JuMP.AbstractModel,
    predictor::Any,
    x::Vector;
    reduced_space::Bool = false,
    kwargs...,
)
    inner_predictor = build_predictor(predictor; kwargs...)
    if reduced_space
        inner_predictor = ReducedSpace(inner_predictor)
    end
    return add_predictor(model, inner_predictor, x)
end

"""
    add_predictor(model::JuMP.AbstractModel, predictor, x::Array; kwargs...)

This method is a helper function for adding `predictor` to `model` when the
input `x` is a multi-dimensional array.

It is equivalent to passing `vec(x)` with the keyword `input_size = size(x)`.

## Example

```jldoctest
julia> using JuMP, MathOptAI, Flux

julia> model = Model();

julia> @variable(model, x[1:4, 1:4]);

julia> predictor = Flux.Chain(Flux.MaxPool((2, 2)), Flux.flatten);

julia> y, formulation = MathOptAI.add_predictor(model, predictor, x);

julia> y
4-element Vector{VariableRef}:
 moai_MaxPool2d[1]
 moai_MaxPool2d[2]
 moai_MaxPool2d[3]
 moai_MaxPool2d[4]

julia> formulation
MaxPool2d((4, 4, 1), (2, 2), (2, 2), (0, 0))
├ variables [4]
│ ├ moai_MaxPool2d[1]
│ ├ moai_MaxPool2d[2]
│ ├ moai_MaxPool2d[3]
│ └ moai_MaxPool2d[4]
└ constraints [4]
  ├ moai_MaxPool2d[1] - max(max(max(x[1,1], x[2,1]), x[1,2]), x[2,2]) = 0
  ├ moai_MaxPool2d[2] - max(max(max(x[3,1], x[4,1]), x[3,2]), x[4,2]) = 0
  ├ moai_MaxPool2d[3] - max(max(max(x[1,3], x[2,3]), x[1,4]), x[2,4]) = 0
  └ moai_MaxPool2d[4] - max(max(max(x[3,3], x[4,3]), x[3,4]), x[4,4]) = 0
```
"""
function add_predictor(
    model::JuMP.AbstractModel,
    predictor,
    x::Array;
    kwargs...,
)
    return add_predictor(
        model,
        predictor,
        vec(x);
        input_size = size(x),
        kwargs...,
    )
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

output_size(p::ReducedSpace, input_size) = output_size(p.predictor, input_size)

struct PaddedArrayView{T} <: AbstractArray{T,3}
    data::Array{T,3}
    padding::Tuple{Int,Int}
end

function Base.getindex(x::PaddedArrayView{T}, i::Int, j::Int, k::Int) where {T}
    i -= x.padding[1]
    j -= x.padding[2]
    if 1 <= i <= size(x.data, 1) && 1 <= j <= size(x.data, 2)
        return x.data[i, j, k]
    end
    return zero(T)
end

"""
    output_size(predictor::AbstractPredictor, input_size::Nothing)
    output_size(predictor::AbstractPredictor, input_size::NTuple{N,Int}) where {N}

Return the output size of `predictor` with an input with shape `input_size`.

If `input_size === nothing`, no information about the input is known. This
function returns an `NTuple{N,Int}` if a static output size based on the
predictor, otherwise returns `nothing`.

## Example

```jldoctest
julia> using MathOptAI

julia> output_size(ReLU(), nothing)

julia> output_size(ReLU(), (2,))
(2,)

julia> output_size(MaxPool2d((3, 3); input_size = (6, 9, 1)), (6, 9, 1))
(2, 3, 1)
```
"""
output_size(::AbstractPredictor, ::Any) = nothing

for file in readdir(joinpath(@__DIR__, "predictors"); join = true)
    if endswith(file, ".jl")
        include(file)
    end
end

include("extension.jl")
include("replace_weights_with_variables.jl")

for sym in names(@__MODULE__; all = true)
    if !Base.isidentifier(sym) || sym in (:eval, :include)
        continue
    elseif startswith("$sym", "_")
        continue
    end
    @eval export $sym
end

end  # module MathOptAI
