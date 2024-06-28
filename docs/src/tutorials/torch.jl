# Copyright (c) 2024: Oscar Dowson and contributors                         #src
# Copyright (c) 2024: Triad National Security, LLC                          #src
#                                                                           #src
# Use of this source code is governed by a BSD-style license that can be    #src
# found in the LICENSE.md file.                                             #src

# # Pytorch

# ## Required packages

ENV["JULIA_CONDAPKG_BACKEND"] = "Current"

using JuMP
using Test

import Ipopt
import MathOptAI
import Plots
import PythonCall

# ## Training a model

# You should probably do this in Python instead.

function train_pytorch_model(filename)
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
        N = torch.normal(torch.zeros(n, 1), torch.ones(n, 1))
        y = -2 * x + x ** 2 + 0.1 * N
        for epoch in range(250):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            model.train()
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: train={(loss.item()):>8f}")

        torch.save(model, file_name)
        """,
        @__MODULE__,
        (; file_name = filename),
    )
end

# ## Read

filename = joinpath(@__DIR__, "model.pt")
train_pytorch_model(filename)
ml_model = MathOptAI.PytorchModel(filename)

# ## JuMP model

model = Model(Ipopt.Optimizer)
set_silent(model)
@variable(model, x)
y = MathOptAI.add_predictor(model, ml_model, [x])
@objective(model, Min, only(y))
X = -2:0.1:2
Y = Float64[]
@constraint(model, c, x == 0.0)
for xi in X
    set_normalized_rhs(c, xi)
    optimize!(model)
    @test is_solved_and_feasible(model)
    push!(Y, objective_value(model))
end
Plots.plot(x -> x * (x - 2), X; label = "Truth")
Plots.plot!(X, Y; label = "Fitted")
