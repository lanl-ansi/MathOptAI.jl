# Copyright (c) 2024: Oscar Dowson and contributors                         #src
# Copyright (c) 2024: Triad National Security, LLC                          #src
#                                                                           #src
# Use of this source code is governed by a BSD-style license that can be    #src
# found in the LICENSE.md file.                                             #src

# # Input Convex Neural Networks with Flux.jl

# This tutorial shows how to embed an input convex neural network (ICNN) model
# from [Flux.jl](https://github.com/FluxML/Flux.jl) into JuMP.

# See [Input Convex Neural Networks with PyTorch](@ref) for this tutorial using
# PyTorch, and see [Input Supermodular Neural Networks with Flux.jl](@ref) for a
# related form of network..

# ## Required packages

# This tutorial requires the following packages:

using JuMP
import Flux
import HiGHS
import Ipopt
import MathOptAI
import Plots
import Random
import SCS

# ## Building the ICNN

# Consider a neural network with the following structure:

# ```math
# \begin{aligned}
# z_1 & = \sigma_1(D_1 x + b_1) \\
# z_k & = \sigma_k(W_{k-1} z_{k-1} + b_k + D_k x), \ \forall k = 2, \ldots, K
# \end{aligned}
# ```
# If the weights $W$ are non-negative and $\sigma$ is a convex activation
# function then the output of the network $z_K$ is convex with respect to $x$.

struct InputConvexNN{T} <: MathOptAI.AbstractPredictor
    D::Vector{Matrix{T}}
    W::Vector{Matrix{T}}
    b::Vector{Vector{T}}
    σ::Vector{Function}
end

Flux.@layer(InputConvexNN, trainable = (D, W, b))

function InputConvexNN(
    dim_in::Int,
    layers::Pair{Int,<:Function}...;
    init = Flux.glorot_uniform,
)
    dims, K = first.(layers), length(layers)
    D = [init(dims[k], dim_in) for k in 1:K]
    W = [init(dims[k], dims[k-1]) for k in 2:K]
    b = [init(dims[k]) for k in 1:K]
    return InputConvexNN(D, W, b, Function[last(l) for l in layers])
end

function (nn::InputConvexNN)(x::AbstractVector)
    z = nn.σ[1].(nn.D[1] * x .+ nn.b[1])
    for k in 2:length(nn.D)
        z = nn.σ[k].(Flux.softplus.(nn.W[k-1]) * z .+ nn.b[k] .+ nn.D[k] * x)
    end
    return z
end

# Here's an example:

chain = InputConvexNN(8, 2 => Flux.relu, 1 => Flux.relu)

#-

chain(rand(8))

# ## Building the Predictor

# We need to implement [`add_predictor`](@ref) for `InputConvexNN` in order to
# be able to embed this network into JuMP.

function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
    nn::InputConvexNN,
    x::Vector;
    kwargs...,
)
    formulation = MathOptAI.PipelineFormulation(predictor, Any[])
    p = MathOptAI.Affine(nn.D[1], nn.b[1])
    z, inner = MathOptAI.add_predictor(model, p, x)
    push!(formulation.layers, inner)
    z, inner = MathOptAI.add_predictor(model, nn.σ[1], z; kwargs...)
    push!(formulation.layers, inner)
    for k in 2:length(nn.D)
        p = MathOptAI.Affine([Flux.softplus.(nn.W[k-1]) nn.D[k]], nn.b[k])
        z, inner = MathOptAI.add_predictor(model, p, [z; x])
        push!(formulation.layers, inner)
        z, inner = MathOptAI.add_predictor(model, nn.σ[k], z; kwargs...)
        push!(formulation.layers, inner)
    end
    return z, formulation
end

# With that, we are now ready to embed these networks into JuMP.

# ## Embed ICNN into JuMP

# Let us build a small ICNN first.

predictor = InputConvexNN(2, 3 => Flux.relu, 1 => Flux.relu)

# We can now embed `predictor` into a JuMP model. We choose to embed the
# `Flux.relu` using [`ReLUSOS1`](@ref):

model = Model()
@variable(model, x[1:2])
config = Dict(Flux.relu => MathOptAI.ReLUSOS1)
z, formulation = MathOptAI.add_predictor(model, predictor, x; config);

#-

z

#-

formulation

# ## Epigraph formulations

# The nice thing about ICNNs is that we can formulate their epigraph and avoid
# adding binary variables to the model. For that, we can use
# [`ReLUEpigraph`](@ref).

# Let's first train a model to predict the relationship $y = x^2$. (Note that
# this is a very basic training loop.)

Random.seed!(1234)
chain = InputConvexNN(1, 10 => Flux.relu, 1 => Flux.relu)
begin
    X = -2.0f0:0.1f0:2.0f0
    optimizer_state = Flux.setup(Flux.Adam(5e-2), chain)
    for epoch in 1:1_000
        _, gradient = Flux.withgradient(chain) do model
            return sum((only(model([x])) - x^2)^2 for x in X)
        end
        Flux.update!(optimizer_state, chain, only(gradient))
    end
end

# Now we can embed the trained network into a JuMP model:

model = Model(HiGHS.Optimizer)
set_silent(model)
@variable(model, x[1:1])
config = Dict(Flux.relu => MathOptAI.ReLUEpigraph)
y, _ = MathOptAI.add_predictor(model, chain, x; config)
@objective(model, Min, only(y))
model

# Because we used the [`ReLUEpigraph`](@ref) predictor, there are no binary or
# integer variables in our model.
#
# Moreover, we can show that the objective value `y` is convex with respect to
# `x`:

x_value, y_value = -2:0.1:2, Float64[]
for xi in x_value
    fix(x[1], xi)
    optimize!(model)
    ## To prove we are solving an LP and not a MIP, require dual solutions.
    assert_is_solved_and_feasible(model; dual = true)
    push!(y_value, objective_value(model))
end
Plots.plot(x_value, y_value; xlabel = "x", ylabel = "y", label = "Trained")
Plots.plot!(x_value, x_value .^ 2; label = "Target", linestyle = :dash)

# ## Conic Formulation

# We can also use [`SoftPlusConicEpigraph`](@ref) in the activation functions.
# The resulting conic formulation can be solved using `SCS` or any other conic
# solver.

Random.seed!(1234)
chain = InputConvexNN(1, 10 => Flux.softplus, 1 => Flux.softplus)
begin
    X = -2.0f0:0.1f0:2.0f0
    optimizer_state = Flux.setup(Flux.Adam(5e-2), chain)
    for epoch in 1:1000
        _, gradient = Flux.withgradient(chain) do model
            return sum((only(model([x])) - x^2)^2 for x in X)
        end
        Flux.update!(optimizer_state, chain, only(gradient))
    end
end

# Next, we embed the neural network using [`SoftPlusConicEpigraph`](@ref).

model = Model(SCS.Optimizer)
set_silent(model)
@variable(model, x[1:1])
config = Dict(Flux.softplus => MathOptAI.SoftPlusConicEpigraph)
y, _ = MathOptAI.add_predictor(model, chain, x; config)
@objective(model, Min, only(y))
model

# Let's draw the same plot to see  the differences in fit with `softplus`.

x_value, y_value = -2:0.1:2, Float64[]
for xi in x_value
    fix(x[1], xi)
    optimize!(model)
    ## To prove we are solving an LP and not a MIP, require dual solutions.
    assert_is_solved_and_feasible(model; dual = true)
    push!(y_value, objective_value(model))
end
Plots.plot(x_value, y_value; xlabel = "x", ylabel = "y", label = "Trained")
Plots.plot!(x_value, x_value .^ 2; label = "Target", linestyle = :dash)

# ## Nonlinear Formulation

# We can also use [`SoftPlusEpigraph`](@ref) in the activation functions.
# The resulting global nonlinear formulation can be solved using `Ipopt` or any
# other nonlinear solver.

model = Model(Ipopt.Optimizer)
set_silent(model)
@variable(model, x[1:1])
config = Dict(Flux.softplus => MathOptAI.SoftPlusEpigraph)
y, _ = MathOptAI.add_predictor(model, chain, x; config)
@objective(model, Min, only(y))
model

# Let's draw the same plot to see  the differences in fit with `softplus`.

x_value, y_value = -2:0.1:2, Float64[]
for xi in x_value
    fix(x[1], xi)
    optimize!(model)
    ## To prove we are solving an LP and not a MIP, require dual solutions.
    assert_is_solved_and_feasible(model; dual = true)
    push!(y_value, objective_value(model))
end
Plots.plot(x_value, y_value; xlabel = "x", ylabel = "y", label = "Trained")
Plots.plot!(x_value, x_value .^ 2; label = "Target", linestyle = :dash)
