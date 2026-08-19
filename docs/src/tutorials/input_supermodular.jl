# Copyright (c) 2024: Oscar Dowson and contributors                         #src
# Copyright (c) 2024: Triad National Security, LLC                          #src
#                                                                           #src
# Use of this source code is governed by a BSD-style license that can be    #src
# found in the LICENSE.md file.                                             #src

# # Input Supermodular Neural Networks with Flux.jl

# This tutorial shows how to embed an input supermodular neural network (ISNN) 
# model from [Flux.jl](https://github.com/FluxML/Flux.jl) into JuMP. The content
# is mostly taken from the paper ["Learning to Solve Bilevel Programs with Binary
# Tender"](https://arxiv.org/pdf/2407.16914).

# This tutorial requires the following packages:

using JuMP
import Flux
import HiGHS
import MathOptAI
import Plots
import Random

# # Building the ISNN

# Consider a neural network with the following structure:

# ```math
# \begin{aligned}
# z_1 & = \sigma(W_1 \tilde{x} + b_1) \\
# z_k & = \sigma(W_k z_{k-1} + b_k + D_k \tilde{x}), \ \forall k = 2, \ldots, K, \\
# \tilde{\phi} & = W_{K + 1} z_{K} + b_{K + 1} + D_{K + 1} \tilde{x}.
# \end{aligned}
# ```

# where $x$ is the input the network and 
# $\tilde{x} := [x^\top, (\mathbf{1} - x)^\top]^\top$. If the weights 
# $W_{1:(K+1)}$ and $D_{2:K}$ are non-negative and $\sigma$ is a convex 
# activation function then the network is said to be supermodular.

# We can implemnt an ISNN in Flux.jl as follows:

struct InputSupermodular{T,F}
    weight_x::Matrix{T}
    weight_z::Matrix{T}
    bias::Vector{T}
    σ::F
end

Flux.@layer(InputSupermodular, trainable = (weight_x, weight_z, bias))

function InputSupermodular(
    ((in_z, in_x), out)::Pair{Tuple{Int,Int},Int},
    σ = identity;
    init = Flux.glorot_uniform,
)
    return InputSupermodular(init(out, in_x), init(out, in_z), init(out), σ)
end

function (c::InputSupermodular)(x::AbstractVector)
    return c.σ.(Flux.softplus.(c.weight_x) * x .+ c.bias), x
end

function (c::InputSupermodular)((z, x)::Tuple)
    return c.σ.(
        Flux.softplus.(c.weight_z) * z .+ Flux.softplus.(c.weight_x) * x .+
        c.bias,
    ),
    x
end

function Base.show(io::IO, l::InputSupermodular)
    m, n = size(l.weight_x)
    print(io, "InputSupermodular((", size(l.weight_z, 2), ", $m) => $n")
    if l.σ != identity
        print(io, ", ", l.σ)
    end
    if l.bias == false
        print(io, "; bias=false")
    end
    print(io, ")")
    return
end

# We can build an `InputSupermodular` layer as follows:

layer = InputSupermodular((8, 8) => 2, Flux.relu)

#-

layer(rand(8))

# We can then define a `Chain` and build an ISNN.

struct InputSupermodularChain{T<:Flux.Chain}
    chain::T
end

InputSupermodularChain(layers...) = InputSupermodularChain(Flux.Chain(layers))

(model::InputSupermodularChain)(x) = first(model.chain(x))

function Base.show(io::IO, l::InputSupermodularChain)
    println(io, "InputSupermodularChain(")
    println.(io, "\t", l.chain)
    println(io, ")")
    return
end

# We also define an `InputConvex` layer for the last layer of the ISNN.

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

chain = InputSupermodularChain(
    InputSupermodular((4, 4) => 4, Flux.relu),
    InputSupermodular((4, 4) => 4, Flux.relu),
    InputConvex((4, 4) => 1),
)

#-

# # Training the network

# We will use the example from the paper to fit the following function:

ϕ(x) = -(min(1 + 2 * abs(x[1] - x[2]), 2) - 2) ^ 2

# We use the following training loop to train our model:

Random.seed!(61)
begin
    optimizer_state = Flux.setup(Flux.Adam(0.05), chain)
    X = [Float32[x1, x2] for x1 in 0:0.05:1, x2 in 0:0.05:1]
    for epoch in 1:1000
        _, gradient = Flux.withgradient(chain) do model
            return sum((only(model([x; 1 .- x])) - ϕ(x))^2 for x in X)
        end
        Flux.update!(optimizer_state, chain, only(gradient))
    end
end

# Let us visualize the true and the fitted function side by side:

p1 = Plots.plot3d(; dpi = 400, size = (800, 400))
Plots.surface!(
    0:0.05:1,
    0:0.05:1,
    (x1, x2) -> ϕ([x1, x2]);
    camera = (105, 15),
    colorbar = false,
)

p2 = Plots.plot3d(; dpi = 400, size = (800, 400))
Plots.surface!(
    0:0.05:1,
    0:0.05:1,
    (x1, x2) -> chain([x1, x2, 1 - x1, 1 - x2]) |> only;
    camera = (105, 15),
    colorbar = false,
)

Plots.plot(p1, p2; layout = (1, 2), size = (800, 400))

# ## Building the predictor

# We need to implement [`build_predictor`](@ref) and [`add_predictor`](@ref) for
# `InputSupermodularChain` in order to be able to embed this network into JuMP.

struct InputSupermodularChainPredictor <: MathOptAI.AbstractPredictor
    p::MathOptAI.Pipeline
end

function MathOptAI.build_predictor(
    predictor::InputSupermodularChain;
    config::Dict = Dict{Any,Any}(),
    kwargs...,
)
    (layer1, layers) = Iterators.peel(predictor.chain)
    p = MathOptAI.Pipeline(
        MathOptAI.Affine(Flux.softplus.(layer1.weight_x), layer1.bias),
        MathOptAI.build_predictor(layer1.σ; config),
    )
    for layer in layers
        weights =
            hcat(Flux.softplus.(layer.weight_z), Flux.softplus.(layer.weight_x))
        push!(p.layers, MathOptAI.Affine(weights, layer.bias))
        push!(p.layers, MathOptAI.build_predictor(layer.σ; config))
    end
    return InputSupermodularChainPredictor(p)
end

function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
    predictor::InputSupermodularChainPredictor,
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

# Now, we are ready to embed the ISNN into a JuMP model.

# ## Embed ISNN into JuMP

# We are going to build a JuMP model with binary decision variables which will 
# be he inputs of the ISNN.

model = Model(HiGHS.Optimizer)
set_silent(model)
@variable(model, x[1:2], Bin)
config = Dict(Flux.relu => MathOptAI.ReLUSOS1)
y, formulation = MathOptAI.add_predictor(model, chain, [x; 1 .- x]; config)

chain

#- 

y

#-

formulation

# We can now solve the model and compare the solutions for both minimization 
# and maximization.:

@objective(model, Max, only(y))
optimize!(model)
println("Maximizer: x* = $(value.(x))")

@objective(model, Min, only(y))
optimize!(model)
println("Minimizer: x* = $(value.(x))")
