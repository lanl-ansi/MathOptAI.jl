# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module GLMTests

using JuMP
using Test

import GLM
import HiGHS
import Ipopt
import MathOptAI

is_test(x) = startswith(string(x), "test_")

function runtests()
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

function test_LinearRegression_GLM()
    num_features = 2
    num_observations = 10
    X = rand(num_observations, num_features)
    θ = rand(num_features)
    Y = X * θ + randn(num_observations)
    model_glm = GLM.lm(X, Y)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, 0 <= x[1:num_features] <= 1)
    @constraint(model, sum(x) == 1.5)
    y = MathOptAI.add_predictor(model, model_glm, x)
    @objective(model, Max, only(y))
    optimize!(model)
    @assert is_solved_and_feasible(model)
    y_star_glm = GLM.predict(model_glm, value.(x)')
    @test isapprox(objective_value(model), y_star_glm; atol = 1e-6)
    return
end

function test_LogisticRegression_GLM()
    num_features = 2
    num_observations = 10
    X = rand(num_observations, num_features)
    θ = rand(num_features)
    Y = X * θ + randn(num_observations) .>= 0
    model_glm = GLM.glm(X, Y, GLM.Bernoulli())
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, 0 <= x[1:num_features] <= 1)
    @constraint(model, sum(x) == 1.5)
    y = MathOptAI.add_predictor(model, model_glm, x)
    @objective(model, Max, only(y))
    optimize!(model)
    @assert is_solved_and_feasible(model)
    y_star_glm = GLM.predict(model_glm, value.(x)')
    @test isapprox(objective_value(model), y_star_glm; atol = 1e-6)
    return
end

end  # module

GLMTests.runtests()
