# Copyright (c) 2024: Oscar Dowson and contributors                         #src
# Copyright (c) 2024: Triad National Security, LLC                          #src
#                                                                           #src
# Use of this source code is governed by a BSD-style license that can be    #src
# found in the LICENSE.md file.                                             #src

# # Input Convex Neural Networks with Flux.jl

# This tutorial shows how to 
# embed an input convex neural 
# network (ICNN) model from 
# [Flux.jl](https://github.com/FluxML/Flux.jl) 
# into JuMP.

# ## Required packages

# This tutorial requires the following packages
using JuMP
using Flux
import MathOptAI

# ## Building the ICNN

# The following custom layer can be used to 
# build ICNNs. This layer has two forward methods.
# One that takes a single input and the other takes 
# a `Tuple`. They both return the result of the 
# forward pass as well as the original input.

struct InputConvex{T,F}
    weight_x::Matrix{T}
    weight_z::Matrix{T}
    bias::Vector{T}
    σ::F
end

Flux.@layer InputConvex trainable=(weight_x, weight_z, bias)

function InputConvex(
    ((in_z, in_x), out)::Pair{Tuple{Int,Int},Int},
    σ = identity;
    init = Flux.glorot_uniform,
)
    return InputConvex(init(out, in_x), init(out, in_z), init(out), σ)
end

function (c::InputConvex)(x)
    return c.σ.(c.weight_x * x .+ c.bias), x
end

function (c::InputConvex)(input::Tuple)
    z, x = input
    return c.σ.(softplus.(c.weight_z) * z .+ c.weight_x * x .+ c.bias), x
end

function Base.show(io::IO, l::InputConvex)
    print(
        io,
        "InputConvex((",
        size(l.weight_z, 2),
        ", ",
        size(l.weight_x, 2),
        ") => ",
        size(l.weight_x, 1),
    )
    l.σ == identity || print(io, ", ", l.σ)
    l.bias == false && print(io, "; bias=false")
    return print(io, ")")
end

# Next, we define a custom `Chain` to build the ICNN.

struct InputConvexChain{T<:Flux.Chain}
    chain::T
end

InputConvexChain(layers...) = InputConvexChain(Chain(layers))

(model::InputConvexChain)(x) = model.chain(x)

function Base.show(io::IO, l::InputConvexChain)
    println(io, "InputConvexChain(")
    println.(io, "\t", l.chain)
    return println(io, ")")
end

# ## Building the Predictor

# We need to implement `build_predictor` and `add_predictor` 
# for `InputConvexChain` in order to be able to embed 
# this network into JuMP.

struct InputConvexChainPredictor <: MathOptAI.AbstractPredictor
    p::Pipeline
end

function MathOptAI.build_predictor(
    predictor::InputConvexChain;
    config::Dict = Dict{Any,Any}(),
    gray_box::Bool = false,
    hessian::Bool = gray_box,
    input_size::Union{Nothing,NTuple} = nothing,
)
    layer1 = first(predictor.chain)
    p = Pipeline(AbstractPredictor[])
    push!(p.layers, Affine(layer1.weight_x, layer1.bias))
    push!(p.layers, build_predictor(layer1.σ; config))
    for layer in predictor.chain[2:end]
        push!(
            p.layers,
            Affine([softplus(layer.weight_z) layer.weight_x], layer.bias),
        )
        push!(p.layers, build_predictor(layer.σ; config))
    end
    return InputConvexChainPredictor(p)
end

function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
    predictor::InputConvexChain,
    x::Vector;
    reduced_space::Bool = false,
    kwargs...,
)::Tuple{<:Vector,<:AbstractFormulation}
    predictor = build_predictor(predictor; kwargs...)
    layer1 = first(predictor.p.layers)
    formulation = PipelineFormulation(predictor, [])
    z, inner_formulation = add_predictor(model, layer1, x)
    push!(formulation.layers, inner_formulation)
    for layer in predictor.p.layers[2:end]
        if layer isa Affine
            z, inner_formulation = add_predictor(model, layer, [z; x])
        else
            z, inner_formulation = add_predictor(model, layer, z)
        end
        push!(formulation.layers, inner_formulation)
    end
    return z, formulation
end

# With that, we are now ready to embed these networks into JuMP.

# ## Embed ICNN into JuMP

# Let us build a small ICNN first.

predictor = InputConvexChain(
    InputConvex((8, 8) => 2, relu),
    InputConvex((2, 8) => 1, relu),
)

# We can embed `predictor` into a JuMP model now.

model = Model();
@variable(model, x[1:8]);

z, formulation =
    add_predictor(model, predictor, x; config = Dict(relu => ReLUSOS1));

z

formulation

# The nice thing about ICNNs is that we can 
# formulate their epigraph and avoid adding binary 
# variables to the model. For that, we can use
# `ReLUEpigraph`.

model = Model();
@variable(model, x[1:8]);

z, formulation =
    add_predictor(model, predictor, x; config = Dict(relu => ReLUEpigraph));

z

formulation
