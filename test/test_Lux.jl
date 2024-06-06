# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module LuxTests

using JuMP
using Test

import ADTypes
import HiGHS
import Lux
import Omelette
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

function test_end_to_end()
    rng = Random.MersenneTwister()
    Random.seed!(rng, 12345)
    x, y = generate_data(rng)
    model = Lux.Chain(Lux.Dense(1 => 16, Lux.relu), Lux.Dense(16 => 1))
    state = train_cpu(
        model,
        x,
        y;
        rng = rng,
        optimizer = Optimisers.Adam(0.03f0),
        epochs = 250,
    )
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x)
    y = Omelette.add_predictor(model, state, [x])
    @constraint(model, only(y) <= 4)
    @objective(model, Min, x)
    optimize!(model)
    @assert is_solved_and_feasible(model)
    @test isapprox(value(x), -1.24; atol = 1e-2)
    return
end

end  # module

LuxTests.runtests()
