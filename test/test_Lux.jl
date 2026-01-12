# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module TestLuxExt

using JuMP
using Test

import HiGHS
import Ipopt
import Lux
import MathOptAI
import Optimisers
import Random
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

function _train_lux_model(model)
    rng = Random.MersenneTwister()
    Random.seed!(rng, 12345)
    x_data, y_data = generate_data(rng)
    parameters, state = Lux.setup(rng, model)
    st_opt = Optimisers.setup(Optimisers.Adam(0.03f0), parameters)
    for epoch in 1:250
        (loss, state), pullback = Zygote.pullback(parameters) do p
            y, new_state = model(x_data, p, state)
            return sum(abs2, y .- y_data), new_state
        end
        gradients = only(pullback((one(loss), nothing)))
        Optimisers.update!(st_opt, parameters, gradients)
    end
    return (model, parameters, state)
end

function test_end_to_end_with_scale()
    state = _train_lux_model(
        Lux.Chain(
            Lux.Scale(1),
            Lux.Dense(1 => 16, Lux.relu),
            Lux.Dense(16 => 1),
        ),
    )
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x)
    y, _ = MathOptAI.add_predictor(
        model,
        state,
        [x];
        config = Dict(Lux.relu => () -> MathOptAI.ReLUBigM(100.0)),
    )
    @constraint(model, only(y) <= 4)
    @objective(model, Min, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    lux_model, lux_p, lux_state = state
    y_val, _ = lux_model(Float32[value(x)], lux_p, lux_state)
    @test isapprox(value.(y), y_val; atol = 1e-2)
    return
end

function test_end_to_end_ReLUBigM()
    state = _train_lux_model(
        Lux.Chain(Lux.Dense(1 => 16, Lux.relu), Lux.Dense(16 => 1)),
    )
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x)
    y, formulation = MathOptAI.add_predictor(
        model,
        state,
        [x];
        config = Dict(Lux.relu => () -> MathOptAI.ReLUBigM(100.0)),
    )
    @constraint(model, only(y) <= 4)
    @objective(model, Min, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    lux_model, lux_p, lux_state = state
    y_val, _ = lux_model(Float32[value(x)], lux_p, lux_state)
    @test isapprox(value.(y), y_val; atol = 1e-2)
    return
end

function test_end_to_end_ReLUQuadratic()
    state = _train_lux_model(
        Lux.Chain(Lux.Dense(1 => 16, Lux.relu), Lux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x)
    y, formulation = MathOptAI.add_predictor(
        model,
        state,
        [x];
        config = Dict(Lux.relu => MathOptAI.ReLUQuadratic),
    )
    # Ipopt needs a starting point to avoid the local minima.
    set_start_value(only(y), 4.0)
    @constraint(model, only(y) <= 4)
    @objective(model, Min, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    lux_model, lux_p, lux_state = state
    y_val, _ = lux_model(Float32[value(x)], lux_p, lux_state)
    @test isapprox(value.(y), y_val; atol = 1e-2)
    return
end

function test_end_to_end_ReLU()
    state = _train_lux_model(
        Lux.Chain(Lux.Dense(1 => 16, Lux.relu), Lux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x)
    y, formulation = MathOptAI.add_predictor(model, state, [x])
    @constraint(model, only(y) <= 4)
    @objective(model, Min, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    lux_model, lux_p, lux_state = state
    y_val, _ = lux_model(Float32[value(x)], lux_p, lux_state)
    @test isapprox(value.(y), y_val; atol = 1e-2)
    return
end

function test_end_to_end_ReLU_reduced_space()
    state = _train_lux_model(
        Lux.Chain(Lux.Dense(1 => 16, Lux.relu), Lux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x)
    y, formulation =
        MathOptAI.add_predictor(model, state, [x]; reduced_space = true)
    @constraint(model, only(y) <= 4)
    @objective(model, Min, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    lux_model, lux_p, lux_state = state
    y_val, _ = lux_model(Float32[value(x)], lux_p, lux_state)
    @test isapprox(value.(y), y_val; atol = 1e-2)
    return
end

function test_end_to_end_Softmax()
    chain = Lux.Chain(Lux.Dense(2 => 3), Lux.softmax)
    rng = Random.MersenneTwister()
    Random.seed!(rng, 12345)
    parameters, state = Lux.setup(rng, chain)
    predictor = (chain, parameters, state)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    y, formulation = MathOptAI.add_predictor(model, predictor, x)
    @constraint(model, x[1] == 1.0)
    @constraint(model, x[2] == 2.0)
    optimize!(model)
    @test is_solved_and_feasible(model)
    y_val, _ = chain(value.(x), parameters, state)
    @test isapprox(value.(y), y_val; atol = 1e-2)
    @test isapprox(sum(value.(y)), 1.0; atol = 1e-2)
    return
end

function test_end_to_end_SoftPlus()
    state = _train_lux_model(
        Lux.Chain(Lux.Dense(1 => 16, Lux.softplus), Lux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x)
    y, formulation = MathOptAI.add_predictor(model, state, [x])
    @constraint(model, only(y) <= 4)
    @objective(model, Min, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    lux_model, lux_p, lux_state = state
    y_val, _ = lux_model(Float32[value(x)], lux_p, lux_state)
    @test isapprox(value.(y), y_val; atol = 1e-2)
    return
end

function test_end_to_end_Sigmoid()
    state = _train_lux_model(
        Lux.Chain(Lux.Dense(1 => 16, Lux.sigmoid), Lux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x)
    y, formulation = MathOptAI.add_predictor(model, state, [x])
    @constraint(model, only(y) <= 4)
    @objective(model, Min, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    lux_model, lux_p, lux_state = state
    y_val, _ = lux_model(Float32[value(x)], lux_p, lux_state)
    @test isapprox(value.(y), y_val; atol = 1e-2)
    return
end

function test_end_to_end_Sigmoid_fast()
    state = _train_lux_model(
        Lux.Chain(Lux.Dense(1 => 16, Lux.sigmoid_fast), Lux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x)
    y, formulation = MathOptAI.add_predictor(model, state, [x])
    @constraint(model, only(y) <= 4)
    @objective(model, Min, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    lux_model, lux_p, lux_state = state
    y_val, _ = lux_model(Float32[value(x)], lux_p, lux_state)
    @test isapprox(value.(y), y_val; atol = 1e-2)
    return
end

function test_end_to_end_Tanh()
    state = _train_lux_model(
        Lux.Chain(Lux.Dense(1 => 16, Lux.tanh), Lux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x)
    y, formulation = MathOptAI.add_predictor(model, state, [x])
    @constraint(model, only(y) <= 4)
    @objective(model, Min, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    lux_model, lux_p, lux_state = state
    y_val, _ = lux_model(Float32[value(x)], lux_p, lux_state)
    @test isapprox(value.(y), y_val; atol = 1e-2)
    return
end

function test_end_to_end_Tanh_fast()
    state = _train_lux_model(
        Lux.Chain(Lux.Dense(1 => 16, Lux.tanh_fast), Lux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x)
    y, formulation = MathOptAI.add_predictor(model, state, [x])
    @constraint(model, only(y) <= 4)
    @objective(model, Min, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    lux_model, lux_p, lux_state = state
    y_val, _ = lux_model(Float32[value(x)], lux_p, lux_state)
    @test isapprox(value.(y), y_val; atol = 1e-2)
    return
end

function test_unsupported_layer()
    layer = Lux.Conv((5, 5), 3 => 7)
    rng = Random.MersenneTwister()
    ml_model = Lux.Chain(layer, layer)
    parameters, state = Lux.setup(rng, ml_model)
    model = Model()
    @variable(model, x[1:2])
    @test_throws(
        ErrorException("Unsupported layer: $layer"),
        MathOptAI.add_predictor(model, (ml_model, parameters, state), x),
    )
    return
end

function test_unsupported_activation()
    chain = Lux.Chain(Lux.Dense(2 => 3, Lux.celu), Lux.softmax)
    rng = Random.MersenneTwister()
    Random.seed!(rng, 12345)
    parameters, state = Lux.setup(rng, chain)
    predictor = (chain, parameters, state)
    model = Model()
    @variable(model, x[1:2])
    @test_throws(
        ErrorException("Unsupported activation function: celu"),
        MathOptAI.add_predictor(model, predictor, x),
    )
    return
end

end  # module

TestLuxExt.runtests()
