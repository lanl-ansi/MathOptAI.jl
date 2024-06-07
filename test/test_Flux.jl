# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module FluxTests

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
    x = range(-2.0, 2.0, n)
    y = -2 .* x .+ x .^ 2 .+ 0.1 .* randn(rng, n)
    return [([xi], yi) for (xi, yi) in zip(x, y)]
end

function _train_lux_model(model)
    rng = Random.MersenneTwister()
    Random.seed!(rng, 12345)
    data = generate_data(rng)
    optim = Flux.setup(Flux.Adam(), model)
    for epoch in 1:1_000
        Flux.train!((m, x, y) -> (only(m(x)) - y)^2, model, data, optim)
    end
    return model
end

function test_end_to_end_ReLUBigM()
    chain = _train_lux_model(
        Flux.Chain(Flux.Dense(1 => 16, Flux.relu), Flux.Dense(16 => 1)),
    )
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x)
    y = MathOptAI.add_predictor(
        model,
        chain,
        [x];
        config = Dict(Flux.relu => MathOptAI.ReLUBigM(100.0)),
    )
    @constraint(model, only(y) <= 4)
    @objective(model, Min, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test isapprox(value(x), -1.24; atol = 1e-2)
    return
end

function test_end_to_end_ReLUQuadratic()
    chain = _train_lux_model(
        Flux.Chain(Flux.Dense(1 => 16, Flux.relu), Flux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x)
    y = MathOptAI.add_predictor(
        model,
        chain,
        [x];
        config = Dict(Flux.relu => MathOptAI.ReLUQuadratic()),
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
    chain = _train_lux_model(
        Flux.Chain(Flux.Dense(1 => 16, Flux.relu), Flux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x)
    y = MathOptAI.add_predictor(model, chain, [x])
    @constraint(model, only(y) <= 4)
    @objective(model, Min, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test isapprox(value(x), -1.24; atol = 1e-2)
    return
end

function test_end_to_end_SoftPlus()
    chain = _train_lux_model(
        Flux.Chain(Flux.Dense(1 => 16, Flux.softplus), Flux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x)
    y = MathOptAI.add_predictor(model, chain, [x])
    @constraint(model, only(y) <= 4)
    @objective(model, Min, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test isapprox(value(x), -1.24; atol = 1e-1)
    return
end

function test_end_to_end_Sigmoid()
    chain = _train_lux_model(
        Flux.Chain(Flux.Dense(1 => 16, Flux.sigmoid), Flux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x)
    y = MathOptAI.add_predictor(model, chain, [x])
    @constraint(model, only(y) <= 4)
    @objective(model, Min, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test isapprox(value(x), -1.24; atol = 1e-1)
    return
end

function test_end_to_end_Tanh()
    chain = _train_lux_model(
        Flux.Chain(Flux.Dense(1 => 16, Flux.tanh), Flux.Dense(16 => 1)),
    )
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x)
    y = MathOptAI.add_predictor(model, chain, [x])
    @constraint(model, only(y) <= 4)
    @objective(model, Min, x)
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test isapprox(value(x), -1.24; atol = 1e-1)
    return
end

end  # module

FluxTests.runtests()
