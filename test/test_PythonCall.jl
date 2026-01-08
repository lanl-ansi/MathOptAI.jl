# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
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
    y, formulation = MathOptAI.add_predictor(
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

function test_model_ReLU_gray_box_config()
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
    @test_throws(
        ErrorException(
            "cannot specify the `config` kwarg if `gray_box = true`",
        ),
        MathOptAI.build_predictor(
            MathOptAI.PytorchModel(filename);
            config = Dict(:ReLU => MathOptAI.ReLUBigM(100)),
            gray_box = true,
        )
    )
    return
end

function test_model_unsupported_layer()
    dir = mktempdir()
    filename = joinpath(dir, "model_RReLU.pt")
    PythonCall.pyexec(
        """
        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.RReLU(),
            torch.nn.Linear(16, 1),
        )

        torch.save(model, filename)
        """,
        @__MODULE__,
        (; filename = filename),
    )
    torch = PythonCall.pyimport("torch")
    layer = torch.nn.RReLU()
    @test_throws(
        ErrorException("unsupported layer: $layer"),
        MathOptAI.build_predictor(MathOptAI.PytorchModel(filename)),
    )
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
    y, formulation = MathOptAI.add_predictor(model, ml_model, x)
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
    y, formulation =
        MathOptAI.add_predictor(model, ml_model, x; reduced_space = true)
    @test num_variables(model) == 1
    @test num_constraints(model; count_variable_in_set_constraints = true) == 0
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test ≈(_evaluate_model(filename, value.(x)), value.(y); atol = 1e-5)
    return
end

function test_model_SoftPlus()
    dir = mktempdir()
    filename = joinpath(dir, "model_SoftPlus.pt")
    PythonCall.pyexec(
        """
        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.Softplus(),
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
    y, formulation = MathOptAI.add_predictor(model, ml_model, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test ≈(_evaluate_model(filename, value.(x)), value.(y); atol = 1e-5)
    return
end

function test_model_SoftPlus_beta()
    dir = mktempdir()
    filename = joinpath(dir, "model_SoftPlus_beta.pt")
    PythonCall.pyexec(
        """
        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.Softplus(beta=0.2),
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
    y, formulation = MathOptAI.add_predictor(model, ml_model, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test ≈(_evaluate_model(filename, value.(x)), value.(y); atol = 1e-5)
    return
end

function test_model_SoftMax()
    dir = mktempdir()
    filename = joinpath(dir, "model_SoftMax.pt")
    PythonCall.pyexec(
        """
        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.Softmax(dim=0),
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
    y, formulation = MathOptAI.add_predictor(model, ml_model, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test ≈(_evaluate_model(filename, value.(x)), value.(y); atol = 1e-5)
    @test ≈(sum(value.(y)), 1.0; atol = 1e-5)
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
    y, formulation = MathOptAI.add_predictor(model, ml_model, x)
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
    @variable(model, x[i in 1:2] == i)
    ml_model = MathOptAI.PytorchModel(filename)
    y, formulation =
        MathOptAI.add_predictor(model, ml_model, x; gray_box = true)
    @test num_variables(model) == 3
    @test num_constraints(model; count_variable_in_set_constraints = true) == 3
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
    @variable(model, x[i in 1:2] == i)
    ml_model = MathOptAI.PytorchModel(filename)
    y, formulation = MathOptAI.add_predictor(
        model,
        ml_model,
        x;
        gray_box = true,
        gray_box_hessian = true,
    )
    @test num_variables(model) == 3
    @test num_constraints(model; count_variable_in_set_constraints = true) == 3
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
    @variable(model, x[i in 1:3] == i)
    ml_model = MathOptAI.PytorchModel(filename)
    y, formulation =
        MathOptAI.add_predictor(model, ml_model, x; gray_box = true)
    @test num_variables(model) == 5
    @test num_constraints(model; count_variable_in_set_constraints = true) == 5
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test ≈(_evaluate_model(filename, value.(x)), value.(y); atol = 1e-5)
    # Reduced-space
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[i in 1:3] == i)
    ml_model = MathOptAI.PytorchModel(filename)
    y, formulation = MathOptAI.add_predictor(
        model,
        ml_model,
        x;
        gray_box = true,
        reduced_space = true,
    )
    @test num_variables(model) == 3
    @test num_constraints(model; count_variable_in_set_constraints = true) == 3
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
    @variable(model, x[i in 1:3] == i)
    ml_model = MathOptAI.PytorchModel(filename)
    y, formulation = MathOptAI.add_predictor(
        model,
        ml_model,
        x;
        gray_box = true,
        gray_box_hessian = true,
    )
    @test num_variables(model) == 5
    @test num_constraints(model; count_variable_in_set_constraints = true) == 5
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test ≈(_evaluate_model(filename, value.(x)), value.(y); atol = 1e-5)
    # Reduced-space
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[i in 1:3] == i)
    ml_model = MathOptAI.PytorchModel(filename)
    y, formulation = MathOptAI.add_predictor(
        model,
        ml_model,
        x;
        gray_box = true,
        reduced_space = true,
    )
    @test num_variables(model) == 3
    @test num_constraints(model; count_variable_in_set_constraints = true) == 3
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test ≈(_evaluate_model(filename, value.(x)), value.(y); atol = 1e-5)
    return
end

function test_model_Sigmoid_last_layer_GrayBox()
    dir = mktempdir()
    filename = joinpath(dir, "model_Sigmoid_last_layer_GrayBox.pt")
    PythonCall.pyexec(
        """
        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.Sigmoid(),
        )

        torch.save(model, filename)
        """,
        @__MODULE__,
        (; filename = filename),
    )
    # Full-space
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[i in 1:3] == i)
    ml_model = MathOptAI.PytorchModel(filename)
    y, formulation =
        MathOptAI.add_predictor(model, ml_model, x; gray_box = true)
    @test num_variables(model) == 19
    @test num_constraints(model; count_variable_in_set_constraints = true) == 19
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test ≈(_evaluate_model(filename, value.(x)), value.(y); atol = 1e-5)
    # Reduced-space
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[i in 1:3] == i)
    ml_model = MathOptAI.PytorchModel(filename)
    y, formulation = MathOptAI.add_predictor(
        model,
        ml_model,
        x;
        gray_box = true,
        reduced_space = true,
    )
    @test num_variables(model) == 3
    @test num_constraints(model; count_variable_in_set_constraints = true) == 3
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test ≈(_evaluate_model(filename, value.(x)), value.(y); atol = 1e-5)
    return
end

function test_vector_nonlinear_oracle_errors()
    @test_throws(
        ErrorException(
            "cannot specify `gray_box = true` if `vector_nonlinear_oracle = true`",
        ),
        MathOptAI.build_predictor(
            MathOptAI.PytorchModel("model.pt");
            gray_box = true,
            vector_nonlinear_oracle = true,
        ),
    )
    @test_throws(
        ErrorException(
            "cannot specify the `config` kwarg if `vector_nonlinear_oracle = true`",
        ),
        MathOptAI.build_predictor(
            MathOptAI.PytorchModel("model.pt");
            config = Dict(:ReLU => MathOptAI.ReLUBigM(100)),
            vector_nonlinear_oracle = true,
        ),
    )
    model = Model()
    x = zeros(10)
    @test_throws(
        ErrorException(
            "cannot construct reduced-space formulation of VectorNonlinearOracle",
        ),
        MathOptAI.add_predictor(
            model,
            MathOptAI.PytorchModel("model.pt"),
            x;
            reduced_space = true,
            vector_nonlinear_oracle = true,
        ),
    )
    return
end

function test_vector_nonlinear_oracle_sigmoid()
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
    @variable(model, x[i in 1:3] == i)
    torch_model = MathOptAI.PytorchModel(filename)
    y, formulation = MathOptAI.add_predictor(
        model,
        torch_model,
        x;
        vector_nonlinear_oracle = true,
    )
    optimize!(model)
    assert_is_solved_and_feasible(model)
    @test isapprox(value.(y), _evaluate_model(filename, value.(x)); atol = 1e-4)
    return
end

function test_vector_nonlinear_oracle_sigmoid_2()
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
    @variable(model, x[i in 1:3] == i)
    @constraint(model, x[1] * x[2]^1.23 <= 4)
    torch_model = MathOptAI.PytorchModel(filename)
    y, formulation = MathOptAI.add_predictor(
        model,
        torch_model,
        x;
        vector_nonlinear_oracle = true,
        device = "cpu",
        hessian = false,
    )
    optimize!(model)
    assert_is_solved_and_feasible(model)
    @test isapprox(value.(y), _evaluate_model(filename, value.(x)); atol = 1e-4)
    return
end

function test_gelu()
    dir = mktempdir()
    filename = joinpath(dir, "model_GELU.pt")
    PythonCall.pyexec(
        """
        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.GELU(approximate='tanh'),
            torch.nn.Linear(16, 2),
        )

        torch.save(model, filename)
        """,
        @__MODULE__,
        (; filename = filename),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[i in 1:3] == i)
    torch_model = MathOptAI.PytorchModel(filename)
    y, _ = MathOptAI.add_predictor(model, torch_model, x)
    optimize!(model)
    assert_is_solved_and_feasible(model)
    @test isapprox(value.(y), _evaluate_model(filename, value.(x)); atol = 1e-4)
    return
end

function test_pr_207()
    dir = mktempdir()
    filename = joinpath(dir, "model_#207.pt")
    PythonCall.pyexec(
        """
        import torch
        activation = torch.nn.ReLU()
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 3),
            activation,
            torch.nn.Linear(3, 4),
            activation,
            torch.nn.Linear(4, 1),
        )
        torch.save(model, filename)
        """,
        @__MODULE__,
        (; filename = filename),
    )
    model = Model()
    @variable(model, x[1:2])
    torch_model = MathOptAI.PytorchModel(filename)
    y, formulation = MathOptAI.add_predictor(model, torch_model, x)
    @test length(formulation.layers) == 5
    @test num_constraints(model, NonlinearExpr, MOI.EqualTo{Float64}) == 7
    return
end

function test_model_LeakyReLU()
    dir = mktempdir()
    filename = joinpath(dir, "model_LeakyReLU.pt")
    PythonCall.pyexec(
        """
        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.LeakyReLU(0.2),
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
    y, formulation = MathOptAI.add_predictor(
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

function test_model_AvgPool2d()
    dir = mktempdir()
    filename = joinpath(dir, "model_AvgPool2d.pt")
    PythonCall.pyexec(
        """
        import torch

        model = torch.nn.Sequential(
            torch.nn.AvgPool2d((2, 2)),
        )

        torch.save(model, filename)
        """,
        @__MODULE__,
        (; filename = filename),
    )
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[h in 1:2, w in 1:4] == w + 4 * (h - 1))
    ml_model = MathOptAI.PytorchModel(filename)
    y, formulation =
        MathOptAI.add_predictor(model, ml_model, vec(x); input_size = (2, 4))
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test value.(y) ≈ [3.5, 5.5]
    return
end

function test_Conv2d()
    dir = mktempdir()
    filename = joinpath(dir, "model_Conv2d.pt")
    PythonCall.pyexec(
        """
        import torch
        conv = torch.nn.Conv2d(1, 2, (2, 2))
        with torch.no_grad():
            conv.weight.copy_(
                torch.arange(8, dtype=conv.weight.dtype).view_as(conv.weight)
            )
            conv.bias.copy_(torch.arange(2))
        model = torch.nn.Sequential(conv)
        torch.save(model, filename)
        """,
        @__MODULE__,
        (; filename = filename),
    )
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[h in 1:2, w in 1:4] == w + 4 * (h - 1))
    ml_model = MathOptAI.PytorchModel(filename)
    y, formulation =
        MathOptAI.add_predictor(model, ml_model, vec(x); input_size = (2, 4))
    optimize!(model)
    @test is_solved_and_feasible(model)
    weight = [3.0 2.0; 1.0 0.0;;;; 7.0 6.0; 5.0 4.0]
    bias = [0.0, 1.0]
    z = value(x)
    y_star = [
        sum(z[:, i:(i+1)] .* weight[2:-1:1, 2:-1:1, 1, j]) + bias[j] for
        j in 1:2 for i in 1:3
    ]
    @test value.(y) ≈ y_star
    torch = PythonCall.pyimport("torch")
    torch_model = torch.load(filename; weights_only = false)
    input = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]])
    y_in = PythonCall.pyconvert(Array, torch_model(input).detach().numpy())
    Y = vec(dropdims(y_in; dims = (1, 3))')
    @test value.(y) ≈ Y
    return
end

function test_model_MaxPool2d()
    dir = mktempdir()
    filename = joinpath(dir, "model_MaxPool2d.pt")
    PythonCall.pyexec(
        """
        import torch

        model = torch.nn.Sequential(
            torch.nn.MaxPool2d((2, 2)),
        )

        torch.save(model, filename)
        """,
        @__MODULE__,
        (; filename = filename),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[h in 1:2, w in 1:4] == w + 4 * (h - 1))
    ml_model = MathOptAI.PytorchModel(filename)
    y, formulation =
        MathOptAI.add_predictor(model, ml_model, vec(x); input_size = (2, 4))
    optimize!(model)
    @test is_solved_and_feasible(model)
    @assert is_solved_and_feasible(model)
    @test value.(y) ≈ [6, 8]
    return
end


function test_large_cnn()
    dir = mktempdir()
    filename = joinpath(dir, "model_large_cnn.pt")
    PythonCall.pyexec(
        """
        import torch
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, (5, 5), padding = 2),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(6, 16, (5, 5), padding = 2),
            torch.nn.AvgPool2d((2, 2)),
            torch.nn.Flatten(0),
            torch.nn.Linear(256, 1),
        )
        torch.save(model, filename)
        """,
        @__MODULE__,
        (; filename = filename),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[i in 1:16, j in 1:16] == i + j)
    cnn = MathOptAI.PytorchModel(filename)
    y, formulation =
        MathOptAI.add_predictor(model, cnn, vec(x); input_size = (16, 16))
    optimize!(model)
    @test is_solved_and_feasible(model)
    torch = PythonCall.pyimport("torch")
    torch_model = torch.load(filename; weights_only = false)
    input = torch.tensor([[fix_value.(x[i, :]) for i in 1:16]])
    y_in = PythonCall.pyconvert(Array, torch_model(input).detach().numpy())
    @test isapprox(only(value(y)), only(y_in); atol = 1e-5)
    return
end

end  # module

TestPythonCallExt.runtests()
