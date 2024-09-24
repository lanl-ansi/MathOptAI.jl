# Copyright (c) 2024: Oscar Dowson and contributors                         #src
# Copyright (c) 2024: Triad National Security, LLC                          #src
#                                                                           #src
# Use of this source code is governed by a BSD-style license that can be    #src
# found in the LICENSE.md file.                                             #src

# # Function fitting with PyTorch

# The purpose of this tutorial is to explain how to embed a neural network model
# from [PyTorch](https://pytorch.org) into JuMP.

# ## Python integration

# MathOptAI uses [PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl)
# to call from Julia into Python.
#
# See [CondaPkg.jl](https://github.com/JuliaPy/CondaPkg.jl) for more control
# over how to link Julia to an existing Python environment. For example, if you
# have an existing Python installation (with PyTorch installed), and it is
# available in the current conda environment, set:
# ```julia
# julia> ENV["JULIA_CONDAPKG_BACKEND"] = "Current"
# ```
# before importing PythonCall.jl. If this Python installation can be found on
# the path, but is not in a conda environment, set:
# ```julia
# julia> ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
# ```

# ## Required packages

# This tutorial requires the following packages

using JuMP
using Test
import Ipopt
import MathOptAI
import Plots

# ## Training a model

# The following script builds and trains a simple neural network in PyTorch.
# For simplicity, we do not evaluate out-of-sample test performance, or use
# a batched data loader. In general, you should train your model in Python,
# and then use `torch.save(model, filename)` to save it to a `.pt` file for
# later use in Julia.

# The model is unimportant, but for this example, we are trying to fit noisy
# observations of the function ``f(x) = x^2 - 2x``.

# In Python, I ran:
# ```python
# #!/usr/bin/python3
# import torch
# model = torch.nn.Sequential(
#     torch.nn.Linear(1, 16),
#     torch.nn.ReLU(),
#     torch.nn.Linear(16, 1),
# )
#
# n = 1024
# x = torch.arange(-2, 2 + 4 / (n - 1), 4 / (n - 1)).reshape(n, 1)
# loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# for epoch in range(100):
#     optimizer.zero_grad()
#     N = torch.normal(torch.zeros(n, 1), torch.ones(n, 1))
#     y = x ** 2 -2 * x + 0.1 * N
#     loss = loss_fn(model(x), y)
#     loss.backward()
#     optimizer.step()
#
# torch.save(model, "model.pt")
# ```

# ## JuMP model

# Our goal for this JuMP model is to load the Neural Network from PyTorch into
# the objective function, and then minimize the objective for different fixed
# values of `x` to recreate the function that the Neural Network has learned to
# approximate.

# First, create a JuMP model:

model = Model(Ipopt.Optimizer)
set_silent(model)
@variable(model, x)

# Then, load the model from PyTorch using [`MathOptAI.PytorchModel`](@ref):

ml_model = MathOptAI.PytorchModel(joinpath(@__DIR__, "model.pt"))
y, _ = MathOptAI.add_predictor(model, ml_model, [x])
@objective(model, Min, only(y))

# Now, visualize the fitted function `y = ml_model(x)` by repeatedly solving the
# optimization problem for different fixed values of `x`:

X, Y = -2:0.1:2, Float64[]
@constraint(model, c, x == 0.0)
for xi in X
    set_normalized_rhs(c, xi)
    optimize!(model)
    @test is_solved_and_feasible(model)
    push!(Y, objective_value(model))
end
Plots.plot(x -> x^2 - 2x, X; label = "Truth", linestype = :dot)
Plots.plot!(X, Y; label = "Fitted")
