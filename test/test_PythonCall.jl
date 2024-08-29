# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module TestPythonCallExt

using JuMP
using Test

import HiGHS
import Ipopt
import MathOptAI
import PythonCall

is_test(x) = startswith(string(x), "test_")

function runtests()
    try
        PythonCall.pyimport("torch")
    catch
        @warn("Skipping PythonCall tests because we cannot import PyTorch.")
        return
    end
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

function _evaluate_model(filename, x)
    torch = PythonCall.pyimport("torch")
    torch_model = torch.load(filename)
    input = torch.tensor(x)
    return PythonCall.pyconvert(Vector, torch_model(input).detach().numpy())
end

function test_model_ReLU()
    dir = mktempdir()
    filename = joinpath(dir, "model_ReLU.pt")
    PythonCall.pyexec(
        """
        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

        torch.save(model, filename)
        """,
        @__MODULE__,
        (; filename = filename),
    )
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[1:1])
    ml_model = MathOptAI.PytorchModel(filename)
    y = MathOptAI.add_predictor(
        model,
        ml_model,
        x;
        config = Dict(:ReLU => MathOptAI.ReLUBigM(100)),
    )
    @objective(model, Min, only(y))
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test ≈(_evaluate_model(filename, value.(x)), value.(y); atol = 1e-5)
    return
end

function test_model_Sigmoid()
    dir = mktempdir()
    filename = joinpath(dir, "model_Sigmoid.pt")
    PythonCall.pyexec(
        """
        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.Sigmoid(),
            torch.nn.Linear(16, 1),
        )

        torch.save(model, filename)
        """,
        @__MODULE__,
        (; filename = filename),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:1])
    ml_model = MathOptAI.PytorchModel(filename)
    y = MathOptAI.add_predictor(model, ml_model, x)
    @objective(model, Min, only(y))
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test ≈(_evaluate_model(filename, value.(x)), value.(y); atol = 1e-5)
    return
end

function test_model_Sigmoid_ReducedSpace()
    dir = mktempdir()
    filename = joinpath(dir, "model_Sigmoid_ReducedSpace.pt")
    PythonCall.pyexec(
        """
        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.Sigmoid(),
            torch.nn.Linear(16, 1),
        )

        torch.save(model, filename)
        """,
        @__MODULE__,
        (; filename = filename),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:1])
    ml_model = MathOptAI.PytorchModel(filename)
    y = MathOptAI.add_predictor(model, ml_model, x; reduced_space = true)
    @objective(model, Min, only(y))
    @test num_variables(model) == 1
    @test num_constraints(model; count_variable_in_set_constraints = true) == 0
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test ≈(_evaluate_model(filename, value.(x)), value.(y); atol = 1e-5)
    return
end

function test_model_Tanh()
    dir = mktempdir()
    filename = joinpath(dir, "model_Tanh.pt")
    PythonCall.pyexec(
        """
        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 1),
        )

        torch.save(model, filename)
        """,
        @__MODULE__,
        (; filename = filename),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:1])
    ml_model = MathOptAI.PytorchModel(filename)
    y = MathOptAI.add_predictor(model, ml_model, x)
    @objective(model, Min, only(y))
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test ≈(_evaluate_model(filename, value.(x)), value.(y); atol = 1e-5)
    return
end

end  # module

TestPythonCallExt.runtests()
