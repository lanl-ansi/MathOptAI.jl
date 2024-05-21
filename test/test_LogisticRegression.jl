# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module LogisticRegressionTests

using JuMP
using Test

import GLM
import Ipopt
import Omelette

is_test(x) = startswith(string(x), "test_")

function runtests()
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

function test_LogisticRegression()
    model = Model()
    @variable(model, x[1:2])
    @variable(model, y[1:1])
    f = Omelette.LogisticRegression([2.0, 3.0])
    Omelette.add_predictor!(model, f, x, y)
    cons = all_constraints(model; include_variable_in_set_constraints = false)
    obj = constraint_object(only(cons))
    @test obj.set == MOI.EqualTo(0.0)
    g = 1.0 / (1.0 + exp(-2.0 * x[1] - 3.0 * x[2])) - y[1]
    @test isequal_canonical(obj.func, g)
    return
end

function test_LogisticRegression_dimension_mismatch()
    model = Model()
    @variable(model, x[1:3])
    @variable(model, y[1:2])
    f = Omelette.LogisticRegression([2.0, 3.0])
    @test size(f) == (1, 2)
    @test_throws DimensionMismatch Omelette.add_predictor!(model, f, x, y[1:1])
    @test_throws DimensionMismatch Omelette.add_predictor!(model, f, x[1:2], y)
    g = Omelette.LogisticRegression([2.0 3.0; 4.0 5.0; 6.0 7.0])
    @test size(g) == (3, 2)
    @test_throws DimensionMismatch Omelette.add_predictor!(model, g, x, y)
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
    model_ml = Omelette.LogisticRegression(model_glm)
    @variable(model, 0 <= x[1:num_features] <= 1)
    @constraint(model, sum(x) == 1.5)
    y = Omelette.add_predictor(model, model_ml, x)
    @objective(model, Max, only(y))
    optimize!(model)
    @assert is_solved_and_feasible(model)
    y_star_glm = GLM.predict(model_glm, value.(x)')
    @test isapprox(objective_value(model), y_star_glm; atol = 1e-6)
    return
end

end

LogisticRegressionTests.runtests()
