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
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test ≈(_evaluate_model(filename, value.(x)), value.(y); atol = 1e-5)
    return
end

function test_model_Tanh_scalar_GrayBox()
    dir = mktempdir()
    filename = joinpath(dir, "model_Tanh_GrayBox.pt")
    PythonCall.pyexec(
        """
        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
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
    @variable(model, x[1:2])
    ml_model = MathOptAI.PytorchModel(filename)
    y = MathOptAI.add_predictor(model, ml_model, x; gray_box = true)
    @test num_variables(model) == 3
    @test num_constraints(model; count_variable_in_set_constraints = true) == 1
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test ≈(_evaluate_model(filename, value.(x)), value.(y); atol = 1e-5)
    return
end

function test_model_Tanh_scalar_GrayBox_hessian()
    dir = mktempdir()
    filename = joinpath(dir, "model_Tanh_GrayBox.pt")
    PythonCall.pyexec(
        """
        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
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
    @variable(model, x[1:2])
    ml_model = MathOptAI.PytorchModel(filename)
    y = MathOptAI.add_predictor(
        model,
        ml_model,
        x;
        gray_box = true,
        gray_box_hessian = true,
    )
    @test num_variables(model) == 3
    @test num_constraints(model; count_variable_in_set_constraints = true) == 1
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test ≈(_evaluate_model(filename, value.(x)), value.(y); atol = 1e-5)
    return
end

function test_model_Tanh_vector_GrayBox()
    dir = mktempdir()
    filename = joinpath(dir, "model_Tanh_vector_GrayBox.pt")
    PythonCall.pyexec(
        """
        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 2),
        )

        torch.save(model, filename)
        """,
        @__MODULE__,
        (; filename = filename),
    )
    # Full-space
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:3])
    ml_model = MathOptAI.PytorchModel(filename)
    y = MathOptAI.add_predictor(model, ml_model, x; gray_box = true)
    @test num_variables(model) == 5
    @test num_constraints(model; count_variable_in_set_constraints = true) == 2
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test ≈(_evaluate_model(filename, value.(x)), value.(y); atol = 1e-5)
    # Reduced-space
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:3])
    ml_model = MathOptAI.PytorchModel(filename)
    y = MathOptAI.add_predictor(
        model,
        ml_model,
        x;
        gray_box = true,
        reduced_space = true,
    )
    @test num_variables(model) == 3
    @test num_constraints(model; count_variable_in_set_constraints = true) == 0
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test ≈(_evaluate_model(filename, value.(x)), value.(y); atol = 1e-5)
    return
end

function test_model_Tanh_vector_GrayBox_hessian()
    dir = mktempdir()
    filename = joinpath(dir, "model_Tanh_vector_GrayBox.pt")
    PythonCall.pyexec(
        """
        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 2),
        )

        torch.save(model, filename)
        """,
        @__MODULE__,
        (; filename = filename),
    )
    # Full-space
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:3])
    ml_model = MathOptAI.PytorchModel(filename)
    y = MathOptAI.add_predictor(
        model,
        ml_model,
        x;
        gray_box = true,
        gray_box_hessian = true,
    )
    @test num_variables(model) == 5
    @test num_constraints(model; count_variable_in_set_constraints = true) == 2
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test ≈(_evaluate_model(filename, value.(x)), value.(y); atol = 1e-5)
    # Reduced-space
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:3])
    ml_model = MathOptAI.PytorchModel(filename)
    y = MathOptAI.add_predictor(
        model,
        ml_model,
        x;
        gray_box = true,
        reduced_space = true,
    )
    @test num_variables(model) == 3
    @test num_constraints(model; count_variable_in_set_constraints = true) == 0
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test ≈(_evaluate_model(filename, value.(x)), value.(y); atol = 1e-5)
    return
end

end  # module

TestPythonCallExt.runtests()
