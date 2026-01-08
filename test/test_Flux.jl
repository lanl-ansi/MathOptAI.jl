# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module TestFluxExt

using JuMP
using Test

import Flux
import HiGHS
import Ipopt
import MathOptAI
import Random

is_test(x) = startswith(string(x), "test_")

function runtests()
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

function generate_data(rng::Random.AbstractRNG, n = 128)
    x = range(-2.0f0, 2.0f0, n)
    y = -2 .* x .+ x .^ 2 .+ 0.1f0 .* randn(rng, n)
    return [([xi], yi) for (xi, yi) in zip(x, y)]
end

function _train_lux_model(model)
    rng = Random.MersenneTwister()
    Random.seed!(rng, 12345)
    data = generate_data(rng)
    optim = Flux.setup(Flux.Adam(0.01f0), model)
    for epoch in 1:250
        Flux.train!((m, x, y) -> (only(m(x)) - y)^2, model, data, optim)
    end
    return model
end

function test_end_to_end_with_scale()
    chain = _train_lux_model(
        Flux.Chain(
            Flux.Scale(1),
            Flux.Dense(1 => 16, Flux.relu),
            Flux.Dense(16 => 1),
        ),
    )
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x == -1.2)
    y, _ = MathOptAI.add_predictor(
        model,
        chain,
        [x];
        config = Dict(Flux.relu => MathOptAI.ReLUBigM(100.0)),
    )
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test isapprox(value.(y), chain(Float32[value(x)]); atol = 1e-2)
    return
end

function test_end_to_end_ReLUBigM()
    chain = _train_lux_model(
        Flux.Chain(Flux.Dense(1 => 16, Flux.relu), Flux.Dense(16 => 1)),
    )
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x == -1.2)
    y, _ = MathOptAI.add_predictor(
        model,
        chain,
        [x];
        config = Dict(Flux.relu => MathOptAI.ReLUBigM(100.0)),
    )
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test isapprox(value.(y), chain(Float32[value(x)]); atol = 1e-2)
    return
end

function test_end_to_end_ReLUQuadratic()
    chain = _train_lux_model(
        Flux.Chain(Flux.Dense(1 => 16, Flux.relu), Flux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x == -1.2)
    y, _ = MathOptAI.add_predictor(
        model,
        chain,
        [x];
        config = Dict(Flux.relu => MathOptAI.ReLUQuadratic()),
    )
    # Ipopt needs a starting point to avoid the local minima.
    set_start_value(only(y), 4.0)
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test isapprox(value.(y), chain(Float32[value(x)]); atol = 1e-2)
    return
end

function test_end_to_end_ReLU()
    chain = _train_lux_model(
        Flux.Chain(Flux.Dense(1 => 16, Flux.relu), Flux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    # set_silent(model)
    @variable(model, x == -1.2)
    y, _ = MathOptAI.add_predictor(model, chain, [x])
    print(model)
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test isapprox(value.(y), chain(Float32[value(x)]); atol = 1e-2)
    return
end

function test_end_to_end_ReLU_reduced_space()
    chain = _train_lux_model(
        Flux.Chain(Flux.Dense(1 => 16, Flux.relu), Flux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x == -1.2)
    y, _ = MathOptAI.add_predictor(model, chain, [x]; reduced_space = true)
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test isapprox(value.(y), chain(Float32[value(x)]); atol = 1e-2)
    return
end

function test_end_to_end_SoftPlus()
    chain = _train_lux_model(
        Flux.Chain(Flux.Dense(1 => 16, Flux.softplus), Flux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x == -1.2)
    y, _ = MathOptAI.add_predictor(model, chain, [x])
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test isapprox(value.(y), chain(Float32[value(x)]); atol = 1e-2)
    return
end

function test_end_to_end_Sigmoid()
    chain = _train_lux_model(
        Flux.Chain(Flux.Dense(1 => 16, Flux.sigmoid), Flux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x == -1.2)
    y, _ = MathOptAI.add_predictor(model, chain, [x])
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test isapprox(value.(y), chain(Float32[value(x)]); atol = 1e-2)
    return
end

function test_end_to_end_Tanh()
    chain = _train_lux_model(
        Flux.Chain(Flux.Dense(1 => 16, Flux.tanh), Flux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x == -1.2)
    y, _ = MathOptAI.add_predictor(model, chain, [x])
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test isapprox(value.(y), chain(Float32[value(x)]); atol = 1e-2)
    return
end

function test_unsupported_layer()
    layer = Flux.Bilinear((5, 5) => 7)
    model = Model()
    @variable(model, x[1:2])
    @test_throws(
        ErrorException("Unsupported layer: $layer"),
        MathOptAI.add_predictor(model, Flux.Chain(layer), x),
    )
    return
end

function test_gray_box_scalar_output()
    chain = Flux.Chain(Flux.Dense(2 => 16, Flux.relu), Flux.Dense(16 => 1))
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_attribute(model, "max_iter", 1)
    @variable(model, 0 <= x[1:2] <= 1)
    y, formulation = MathOptAI.add_predictor(
        model,
        chain,
        x;
        gray_box = true,
        reduced_space = true,
    )
    @objective(model, Max, only(y))
    optimize!(model)
    @test termination_status(model) == ITERATION_LIMIT
    @test isapprox(value.(y), chain(Float32.(value.(x))); atol = 1e-2)
    y, formulation = MathOptAI.add_predictor(model, chain, x; gray_box = true)
    @test y isa Vector{VariableRef}
    config = Dict(Flux.relu => MathOptAI.ReLU())
    @test_throws(
        ErrorException(
            "cannot specify the `config` kwarg if `gray_box = true`",
        ),
        MathOptAI.add_predictor(model, chain, x; gray_box = true, config),
    )
    return
end

function test_gray_box_scalar_output_hessian()
    chain = Flux.Chain(Flux.Dense(2 => 16, Flux.relu), Flux.Dense(16 => 1))
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_attribute(model, "max_iter", 1)
    @variable(model, 0 <= x[1:2] <= 1)
    y, formulation = MathOptAI.add_predictor(
        model,
        chain,
        x;
        gray_box = true,
        gray_box_hessian = true,
        reduced_space = true,
    )
    @objective(model, Max, only(y))
    optimize!(model)
    @test termination_status(model) == ITERATION_LIMIT
    @test isapprox(value.(y), chain(Float32.(value.(x))); atol = 1e-2)
    return
end

function test_gray_box_vector_output()
    chain = Flux.Chain(Flux.Dense(3 => 16, Flux.relu), Flux.Dense(16 => 2))
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_attribute(model, "max_iter", 1)
    @variable(model, 0 <= x[1:3] <= 1)
    y, formulation = MathOptAI.add_predictor(
        model,
        chain,
        x;
        gray_box = true,
        reduced_space = true,
    )
    @test length(y) == 2
    @objective(model, Max, sum(y))
    optimize!(model)
    @test termination_status(model) == ITERATION_LIMIT
    @test isapprox(value.(y), chain(Float32.(value.(x))); atol = 1e-2)
    return
end

function test_gray_box_vector_output_hessian()
    chain = Flux.Chain(Flux.Dense(3 => 16, Flux.relu), Flux.Dense(16 => 2))
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_attribute(model, "max_iter", 1)
    @variable(model, 0 <= x[1:3] <= 1)
    y, formulation = MathOptAI.add_predictor(
        model,
        chain,
        x;
        gray_box = true,
        gray_box_hessian = true,
        reduced_space = true,
    )
    @test length(y) == 2
    @objective(model, Max, sum(y))
    optimize!(model)
    @test termination_status(model) in (LOCALLY_SOLVED, ITERATION_LIMIT)
    @test isapprox(value.(y), chain(Float32.(value.(x))); atol = 1e-2)
    return
end

function test_end_to_end_Softmax()
    chain = Flux.Chain(Flux.Dense(2 => 3), Flux.softmax)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[i in 1:2] == i)
    y, _ = MathOptAI.add_predictor(model, chain, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    y_val = chain(Float32.(value.(x)))
    @test isapprox(value.(y), y_val; atol = 1e-2)
    @test isapprox(sum(value.(y)), 1.0; atol = 1e-2)
    return
end

function test_unsupported_activation()
    chain = Flux.Chain(Flux.Dense(2 => 3, Flux.celu), Flux.softmax)
    model = Model()
    @variable(model, x[1:2])
    @test_throws(
        ErrorException("Unsupported activation function: celu"),
        MathOptAI.add_predictor(model, chain, x),
    )
    return
end

function test_vector_nonlinear_oracle_errors()
    chain = Flux.Chain(Flux.Dense(3 => 16, Flux.sigmoid), Flux.Dense(16 => 2))
    @test_throws(
        ErrorException(
            "cannot specify `gray_box = true` if `vector_nonlinear_oracle = true`",
        ),
        MathOptAI.build_predictor(
            chain;
            gray_box = true,
            vector_nonlinear_oracle = true,
        ),
    )
    @test_throws(
        ErrorException(
            "cannot specify the `config` kwarg if `vector_nonlinear_oracle = true`",
        ),
        MathOptAI.build_predictor(
            chain;
            config = Dict(Flux.relu => MathOptAI.ReLUBigM(100.0)),
            vector_nonlinear_oracle = true,
        ),
    )
    return
end

function test_vector_nonlinear_oracle_sigmoid()
    chain = Flux.Chain(Flux.Dense(3 => 16, Flux.sigmoid), Flux.Dense(16 => 2))
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[i in 1:3] == i)
    y, formulation =
        MathOptAI.add_predictor(model, chain, x; vector_nonlinear_oracle = true)
    optimize!(model)
    assert_is_solved_and_feasible(model)
    @test isapprox(value.(y), chain(Float32.(value.(x))); atol = 1e-4)
    return
end

function test_vector_nonlinear_oracle_sigmoid_2()
    chain = Flux.Chain(Flux.Dense(3 => 16, Flux.sigmoid), Flux.Dense(16 => 2))
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[i in 1:3] == i)
    @constraint(model, x[1] * x[2]^1.23 <= 4)
    y, formulation = MathOptAI.add_predictor(
        model,
        chain,
        x;
        hessian = false,
        vector_nonlinear_oracle = true,
    )
    optimize!(model)
    assert_is_solved_and_feasible(model)
    @test isapprox(value.(y), chain(Float32.(value.(x))); atol = 1e-4)
    return
end

function test_vector_nonlinear_oracle_sigmoid_reduced_space_error()
    chain = Flux.Chain(Flux.Dense(3 => 16, Flux.sigmoid), Flux.Dense(16 => 2))
    model = Model()
    @variable(model, x[i in 1:3] == i)
    @test_throws(
        ErrorException(
            "cannot construct reduced-space formulation of VectorNonlinearOracle",
        ),
        MathOptAI.add_predictor(
            model,
            chain,
            x;
            reduced_space = true,
            vector_nonlinear_oracle = true,
        ),
    )
    return
end

function test_AvgPool2d_against_flux()
    model = Model()
    for (H, W, C, kernel, pad, stride) in [
        (16, 16, 1, (5, 5), 0, (1, 1)),
        (16, 16, 1, (5, 5), 1, (1, 1)),
        (16, 16, 1, (5, 5), 2, (1, 1)),
        (16, 16, 1, (5, 5), 1, (1, 1)),
        (16, 16, 2, (5, 5), 1, (1, 1)),
        (16, 16, 2, (2, 3), 1, (1, 1)),
        (3, 5, 2, (2, 3), 1, (1, 1)),
        (20, 20, 2, (4, 4), 0, (4, 4)),
    ]
        x = rand(Float32, H, W, C, 1);
        f = Flux.MeanPool(kernel; pad, stride)
        g = MathOptAI.AvgPool2d(
            kernel;
            input_size = size(x)[1:3],
            padding = f.pad[1:2],
            stride = f.stride,
        )
        A, B = f(x), g(model, vec(x))
        @test size(A)[1:3] == size(B)
        @test maximum(abs, A - B) < 1e-6
    end
    return
end

function test_Conv2d_against_flux()
    model = Model()
    for (H, W, C, kernel, pad, stride) in [
        (16, 16, 1 => 1, (5, 5), 0, (1, 1)),
        (16, 16, 1 => 1, (5, 5), 1, (1, 1)),
        (16, 16, 1 => 1, (5, 5), 2, (1, 1)),
        (16, 16, 1 => 5, (5, 5), 1, (1, 1)),
        (16, 16, 2 => 2, (5, 5), 1, (1, 1)),
        (16, 16, 2 => 2, (2, 3), 1, (1, 1)),
        (3, 5, 2 => 2, (2, 3), 1, (1, 1)),
        (20, 20, 2 => 3, (4, 4), 0, (4, 4)),
    ]
        x = rand(Float32, H, W, first(C), 1);
        f = Flux.Conv(kernel, C, identity; pad, stride)
        g = MathOptAI.Conv2d(
            f.weight,
            f.bias;
            input_size = size(x)[1:3],
            padding = f.pad[1:2],
            stride = f.stride,
        )
        A, B = f(x), g(model, vec(x))
        @test size(A)[1:3] == size(B)
        @test maximum(abs, A - B) < 1e-6
    end
    return
end

function test_MaxPool_against_flux()
    model = Model()
    for (H, W, C, kernel, pad, stride) in [
        (16, 16, 1, (5, 5), 0, (1, 1)),
        (16, 16, 1, (5, 5), 1, (1, 1)),
        (16, 16, 1, (5, 5), 2, (1, 1)),
        (16, 16, 1, (5, 5), 1, (1, 1)),
        (16, 16, 2, (5, 5), 1, (1, 1)),
        (16, 16, 2, (2, 3), 1, (1, 1)),
        (3, 5, 2, (2, 3), 1, (1, 1)),
        (20, 20, 2, (4, 4), 0, (4, 4)),
    ]
        x = rand(Float32, H, W, C, 1);
        f = Flux.MaxPool(kernel; pad, stride)
        g = MathOptAI.MaxPool2d(
            kernel;
            input_size = size(x)[1:3],
            padding = f.pad[1:2],
            stride = f.stride,
        )
        A, B = f(x), g(model, vec(x))
        @test size(A)[1:3] == size(B)
        @test maximum(abs, A - B) < 1e-6
    end
    return
end

function test_flux_large_cnn()
    cnn = Chain(                            # (16, 16, 1, 1)
        Conv((5, 5), 1=>6, relu, pad = 2),  # -> (16, 16, 6, 1)
        MaxPool((2, 2)),                    # -> (8, 8, 6, 1)
        Conv((5, 5), 6=>16, relu, pad = 2), # -> (8, 8, 16, 1)
        MaxPool((2, 2)),                    # -> (4, 4, 16, 1)
        Flux.flatten,                       # -> (256,)
        Dense(256 => 120, relu),            # -> (120,)
        Dense(120 => 84, relu),             # -> (84,)
        Dense(84 => 10),                    # -> (10,)
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[i in 1:16, j in 1:16] == i + j)
    y, formulation =
        MathOptAI.add_predictor(model, cnn, vec(x); input_size = (16, 16))
    @test length(y) == 10
    optimize!(model)
    assert_is_solved_and_feasible(model)
    @test maximum(
        abs,
        value(y) - cnn(convert.(Float32, reshape(fix_value.(x), 16, 16, 1, 1))),
    ) <= 1e-5
    return
end

end  # module

TestFluxExt.runtests()
