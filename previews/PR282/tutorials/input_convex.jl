# Copyright (c) 2024: Oscar Dowson and contributors                         #src
# Copyright (c) 2024: Triad National Security, LLC                          #src
#                                                                           #src
# Use of this source code is governed by a BSD-style license that can be    #src
# found in the LICENSE.md file.                                             #src

# # Input Convex Neural Networks with Flux.jl

# This tutorial shows how to embed an input convex neural network (ICNN) model
# from [Flux.jl](https://github.com/FluxML/Flux.jl) into JuMP.

# ## Required packages

# This tutorial requires the following packages:

using JuMP
import Flux
import HiGHS
import MathOptAI
import Plots

# ## Building the ICNN

# The following custom layer can be used to build ICNNs. This layer has two
# forward methods. One that takes a single input and the other takes  a `Tuple`.
# They both return the result of the forward pass as well as the original input.

struct InputConvex{T,F}
    weight_x::Matrix{T}
    weight_z::Matrix{T}
    bias::Vector{T}
    σ::F
end

Flux.@layer(InputConvex, trainable = (weight_x, weight_z, bias))

function InputConvex(
    ((in_z, in_x), out)::Pair{Tuple{Int,Int},Int},
    σ = identity;
    init = Flux.glorot_uniform,
)
    return InputConvex(init(out, in_x), init(out, in_z), init(out), σ)
end

function (c::InputConvex)(x::AbstractVector)
    return c.σ.(c.weight_x * x .+ c.bias), x
end

function (c::InputConvex)((z, x)::Tuple)
    return c.σ.(Flux.softplus.(c.weight_z) * z .+ c.weight_x * x .+ c.bias), x
end

function Base.show(io::IO, l::InputConvex)
    m, n = size(l.weight_x)
    print(io, "InputConvex((", size(l.weight_z, 2), ", $m) => $n")
    if l.σ != identity
        print(io, ", ", l.σ)
    end
    if l.bias == false
        print(io, "; bias=false")
    end
    print(io, ")")
    return
end

# Here's an example:

layer = InputConvex((8, 8) => 2, Flux.relu)

#-

layer(rand(8))

# Next, we define a custom `Chain` to build the ICNN.

struct InputConvexChain{T<:Flux.Chain}
    chain::T
end

InputConvexChain(layers...) = InputConvexChain(Flux.Chain(layers))

(model::InputConvexChain)(x) = model.chain(x)

function Base.show(io::IO, l::InputConvexChain)
    println(io, "InputConvexChain(")
    println.(io, "\t", l.chain)
    println(io, ")")
    return
end

# Here's an example:

chain = InputConvexChain(
    InputConvex((8, 8) => 2, Flux.relu),
    InputConvex((2, 8) => 1, Flux.relu),
)

#-

chain(rand(8))

# ## Building the Predictor

# We need to implement [`build_predictor`](@ref) and [`add_predictor`](@ref) for
# `InputConvexChain` in order to be able to embed this network into JuMP.

struct InputConvexChainPredictor <: MathOptAI.AbstractPredictor
    p::MathOptAI.Pipeline
end

function MathOptAI.build_predictor(
    predictor::InputConvexChain;
    config::Dict = Dict{Any,Any}(),
    kwargs...,
)
    layer1 = first(predictor.chain)
    p = MathOptAI.Pipeline(
        MathOptAI.Affine(layer1.weight_x, layer1.bias),
        MathOptAI.build_predictor(layer1.σ; config),
    )
    for layer in predictor.chain[2:end]
        weights = hcat(Flux.softplus(layer.weight_z), layer.weight_x)
        push!(p.layers, MathOptAI.Affine(weights, layer.bias))
        push!(p.layers, MathOptAI.build_predictor(layer.σ; config))
    end
    return InputConvexChainPredictor(p)
end

function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
    predictor::InputConvexChainPredictor,
    x::Vector;
    kwargs...,
)
    layers = predictor.p.layers
    z, inner = MathOptAI.add_predictor(model, first(layers), x)
    formulation = MathOptAI.PipelineFormulation(predictor, Any[inner])
    for layer in layers[2:end]
        z, inner = if layer isa MathOptAI.Affine
            MathOptAI.add_predictor(model, layer, [z; x])
        else
            MathOptAI.add_predictor(model, layer, z)
        end
        push!(formulation.layers, inner)
    end
    return z, formulation
end

# With that, we are now ready to embed these networks into JuMP.

# ## Embed ICNN into JuMP

# Let us build a small ICNN first.

predictor = InputConvexChain(
    InputConvex((8, 8) => 2, Flux.relu),
    InputConvex((2, 8) => 1, Flux.relu),
)

# We can embed `predictor` into a JuMP model now.

model = Model()
@variable(model, x[1:8])
z, formulation = MathOptAI.add_predictor(
    model,
    predictor,
    x;
    config = Dict(Flux.relu => MathOptAI.ReLUSOS1),
);

#-

z

#-

formulation

# ## Epigraph formulations

# The nice thing about ICNNs is that we can formulate their epigraph and avoid
# adding binary variables to the model. For that, we can use
# [`ReLUEpigraph`](@ref).

chain = InputConvexChain(
    InputConvex((1, 1) => 3, Flux.relu),
    InputConvex((3, 1) => 1, Flux.relu),
)
model = Model(HiGHS.Optimizer)
set_silent(model)
@variable(model, x[1:1])
config = Dict(Flux.relu => MathOptAI.ReLUEpigraph)
y, _ = MathOptAI.add_predictor(model, chain, x; config)
@objective(model, Min, only(y))
x_value, y_value = -20:20, Float64[]
for xi in x_value
    fix(x[1], xi)
    optimize!(model)
    assert_is_solved_and_feasible(model)
    push!(y_value, objective_value(model))
end
Plots.plot(x_value, y_value)

# As expected, the value of `y` is convex with respect to `x`.
