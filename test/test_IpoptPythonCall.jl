# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module TestIpoptPythonCallExt

using JuMP
using Test

import Ipopt
import MathOptAI
import PythonCall

is_test(x) = startswith(string(x), "test_")

function runtests()
    # If we're running the tests locally, allow skipping Python tests
    if get(ENV, "CI", "false") == "false"
        try
            PythonCall.pyimport("torch")
        catch
            @warn("Skipping PythonCall tests because we cannot import PyTorch.")
            return
        end
    end
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

function _evaluate_model(filename, x)
    torch = PythonCall.pyimport("torch")
    torch_model = torch.load(filename; weights_only = false)
    input = torch.tensor(x)
    return PythonCall.pyconvert(Vector, torch_model(input).detach().numpy())
end

function test_sigmoid()
    dir = mktempdir()
    filename = joinpath(dir, "model_Sigmoid.pt")
    PythonCall.pyexec(
        """
        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.Sigmoid(),
            torch.nn.Linear(16, 2),
        )

        torch.save(model, filename)
        """,
        @__MODULE__,
        (; filename = filename),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[i = 1:3] == i)
    torch_model = MathOptAI.PytorchModel(filename)
    predictor = MathOptAI.VectorNonlinearOracle(torch_model)
    y, formulation = MathOptAI.add_predictor(model, predictor, x)
    optimize!(model)
    assert_is_solved_and_feasible(model)
    @test isapprox(value.(y), _evaluate_model(filename, value.(x)); atol = 1e-4)
    return
end

function test_sigmoid_2()
    dir = mktempdir()
    filename = joinpath(dir, "model_Sigmoid2.pt")
    PythonCall.pyexec(
        """
        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.Sigmoid(),
            torch.nn.Linear(16, 2),
        )

        torch.save(model, filename)
        """,
        @__MODULE__,
        (; filename = filename),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[i = 1:3] == i)
    @constraint(model, x[1] * x[2]^1.23 <= 4)
    torch_model = MathOptAI.PytorchModel(filename)
    predictor = MathOptAI.VectorNonlinearOracle(torch_model)
    y, formulation = MathOptAI.add_predictor(model, predictor, x)
    optimize!(model)
    assert_is_solved_and_feasible(model)
    @test isapprox(value.(y), _evaluate_model(filename, value.(x)); atol = 1e-4)
    return
end

end  # module

TestIpoptPythonCallExt.runtests()
