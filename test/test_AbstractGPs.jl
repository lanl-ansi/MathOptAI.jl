# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module TestAbstractGPsExt

using JuMP
using Test

import AbstractGPs
import Ipopt
import MathOptAI

is_test(x) = startswith(string(x), "test_")

function runtests()
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

function test_univariate()
    x = [-2, -1, 2]
    y = x .^ 2
    fx = AbstractGPs.GP(AbstractGPs.Matern32Kernel())(x, 0.1)
    p_fx = AbstractGPs.posterior(fx, y)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, -2 <= x_input[1:1] <= 2, start = 1)
    predictor = MathOptAI.Quantile(p_fx, [0.1, 0.9])
    y_output, formulation = MathOptAI.add_predictor(model, predictor, x_input)
    @objective(model, Max, y_output[2] - y_output[1])
    optimize!(model)
    @test is_solved_and_feasible(model)
    x_sol = value.(x_input)
    y_sol = value.(y_output)
    μ, σ² = only.(AbstractGPs.mean_and_var(p_fx(x_sol)))
    λ = 1.2815515655446001
    y_target = [μ - λ * sqrt(σ²), μ + λ * sqrt(σ²)]
    @test ≈(y_sol, y_target; atol = 1e-4)
    return
end

function test_multivariate()
    x = [-1.2 -2; -1 -1; 2 1.5]
    y = vec(prod(x; dims = 2))
    kernel = AbstractGPs.Matern32Kernel()
    fx = AbstractGPs.GP(kernel)(AbstractGPs.RowVecs(x), 0.1)
    p_fx = AbstractGPs.posterior(fx, y)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, -2 <= x_input[1:2] <= 2, start = 1)
    predictor = MathOptAI.Quantile(p_fx, [0.1, 0.9])
    y_output, formulation = MathOptAI.add_predictor(model, predictor, x_input)
    @objective(model, Max, y_output[2] - y_output[1])
    optimize!(model)
    @test is_solved_and_feasible(model)
    x_sol = AbstractGPs.RowVecs(value.(x_input)')
    y_sol = value.(y_output)
    μ, σ² = only.(AbstractGPs.mean_and_var(p_fx(x_sol)))
    λ = 1.2815515655446001
    y_target = [μ - λ * sqrt(σ²), μ + λ * sqrt(σ²)]
    @test ≈(y_sol, y_target; atol = 1e-4)
    return
end

end  # module

TestAbstractGPsExt.runtests()
