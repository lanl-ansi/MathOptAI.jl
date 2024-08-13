# Copyright (c) 2024: Oscar Dowson and contributors                         #src
# Copyright (c) 2024: Triad National Security, LLC                          #src
#                                                                           #src
# Use of this source code is governed by a BSD-style license that can be    #src
# found in the LICENSE.md file.                                             #src

# # Pytorch

# The purpose of this tutorial is to explain how to embed a neural network model
# from [Pytorch](https://pytorch.org) into JuMP.

# ## Python integration

# This tutorial uses [PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl)
# to call from Julia into Python.
#
# See [CondaPkg.jl](https://github.com/JuliaPy/CondaPkg.jl) for more control
# over how to link Julia to an existing Python environment. For example, if you
# have an existing Python installation (with Pytorch installed), and it is
# available in the current conda environment, set:
# ```julia
# julia> ENV["JULIA_CONDAPKG_BACKEND"] = "Current"
# ```

# ## Required packages

# This tutorial requires the following packages

using JuMP
using Test
import Ipopt
import MathOptAI
import Plots
import PythonCall

# ## Training a model

# The following script builds and trains a simple neural network in Pytorch.
# For simplicity, we do not evaluate out-of-sample test performance, or use
# a batched data loader. In general, you should train your model in Python,
# and then use `torch.save(model, filename)` to save it to a `.pt` file for
# later use in Julia.

filename = joinpath(@__DIR__, "model.pt")
PythonCall.pyexec(
    """
    import torch

    model = torch.nn.Sequential(
        torch.nn.Linear(1, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
    )

    n = 1024
    x = torch.arange(-2, 2 + 4 / (n - 1), 4 / (n - 1)).reshape(n, 1)
    for epoch in range(100):
        N = torch.normal(torch.zeros(n, 1), torch.ones(n, 1))
        y = -2 * x + x ** 2 + 0.1 * N
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train={(loss.item()):>8f}")

    torch.save(model, filename)
    """,
    @__MODULE__,
    (; filename = filename),
)

# ## JuMP model

# Load a model from Pytorch using [`MathOptAI.PytorchModel`](@ref).

model = Model(Ipopt.Optimizer)
set_silent(model)
@variable(model, x)
ml_model = MathOptAI.PytorchModel(filename)
y = MathOptAI.add_predictor(model, ml_model, [x])
@objective(model, Min, only(y))
X, Y = -2:0.1:2, Float64[]
@constraint(model, c, x == 0.0)
for xi in X
    set_normalized_rhs(c, xi)
    optimize!(model)
    @test is_solved_and_feasible(model)
    push!(Y, objective_value(model))
end
Plots.plot(x -> x * (x - 2), X; label = "Truth", linestype = :dot)
Plots.plot!(X, Y; label = "Fitted")
