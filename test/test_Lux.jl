# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module LuxTests

using JuMP
using Test

import ADTypes
import HiGHS
import Ipopt
import Lux
import MathOptAI
import Optimisers
import Random
import Statistics
import Zygote

is_test(x) = startswith(string(x), "test_")

function runtests()
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

function generate_data(rng::Random.AbstractRNG, n = 128)
    x = range(-2.0, 2.0, n)
    y = -2 .* x .+ x .^ 2 .+ 0.1 .* randn(rng, n)
    return reshape(collect(x), (1, n)), reshape(y, (1, n))
end

function loss_function_mse(model, ps, state, (input, output))
    y_pred, updated_state = Lux.apply(model, input, ps, state)
    loss = Statistics.mean(abs2, y_pred .- output)
    return loss, updated_state, ()
end

function train_cpu(
    model,
    input,
    output;
    loss_function::Function = loss_function_mse,
    vjp = ADTypes.AutoZygote(),
    rng,
    optimizer,
    epochs::Int,
)
    state = Lux.Experimental.TrainState(rng, model, optimizer)
    data = (input, output) .|> Lux.cpu_device()
    for epoch in 1:epochs
        grads, loss, stats, state =
            Lux.Experimental.compute_gradients(vjp, loss_function, data, state)
        state = Lux.Experimental.apply_gradients!(state, grads)
    end
    return state
end

function _train_lux_model(model)
    rng = Random.MersenneTwister()
    Random.seed!(rng, 12345)
    x, y = generate_data(rng)
    state = train_cpu(
        model,
        x,
        y;
        rng = rng,
        optimizer = Optimisers.Adam(0.03f0),
        epochs = 250,
    )
    return state
end

function test_end_to_end_ReLUBigM()
    state = _train_lux_model(
        Lux.Chain(Lux.Dense(1 => 16, Lux.relu), Lux.Dense(16 => 1)),
    )
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x)
    y = MathOptAI.add_predictor(
        model,
        state,
        [x],
        MathOptAI.ReplaceConfig(MathOptAI.ReLU() => MathOptAI.ReLUBigM(100.0)),
    )
    @constraint(model, only(y) <= 4)
    @objective(model, Min, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test isapprox(value(x), -1.24; atol = 1e-2)
    return
end

function test_end_to_end_ReLUQuadratic()
    state = _train_lux_model(
        Lux.Chain(Lux.Dense(1 => 16, Lux.relu), Lux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x)
    y = MathOptAI.add_predictor(
        model,
        state,
        [x],
        MathOptAI.ReplaceConfig(MathOptAI.ReLU() => MathOptAI.ReLUQuadratic()),
    )
    # Ipopt needs a starting point to avoid the local minima.
    set_start_value(only(y), 4.0)
    @constraint(model, only(y) <= 4)
    @objective(model, Min, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test isapprox(value(x), -1.24; atol = 1e-2)
    return
end

function test_end_to_end_ReLU()
    state = _train_lux_model(
        Lux.Chain(Lux.Dense(1 => 16, Lux.relu), Lux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x)
    y = MathOptAI.add_predictor(model, state, [x])
    @constraint(model, only(y) <= 4)
    @objective(model, Min, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test isapprox(value(x), -1.24; atol = 1e-2)
    return
end

function test_end_to_end_SoftPlus()
    state = _train_lux_model(
        Lux.Chain(Lux.Dense(1 => 16, Lux.softplus), Lux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x)
    y = MathOptAI.add_predictor(model, state, [x])
    @constraint(model, only(y) <= 4)
    @objective(model, Min, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test isapprox(value(x), -1.24; atol = 1e-1)
    return
end

function test_end_to_end_Sigmoid()
    state = _train_lux_model(
        Lux.Chain(Lux.Dense(1 => 16, Lux.sigmoid), Lux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x)
    y = MathOptAI.add_predictor(model, state, [x])
    @constraint(model, only(y) <= 4)
    @objective(model, Min, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test isapprox(value(x), -1.24; atol = 1e-1)
    return
end

function test_end_to_end_Tanh()
    state = _train_lux_model(
        Lux.Chain(Lux.Dense(1 => 16, Lux.tanh), Lux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x)
    y = MathOptAI.add_predictor(model, state, [x])
    @constraint(model, only(y) <= 4)
    @objective(model, Min, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test isapprox(value(x), -1.24; atol = 1e-1)
    return
end

end  # module

LuxTests.runtests()
