# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module TestDecisionTreeExt

using JuMP
using Test

import DecisionTree
import HiGHS
import MathOptAI
import Random

is_test(x) = startswith(string(x), "test_")

function runtests()
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

function test_DecisionTree()
    truth(x::Vector) = x[1] <= 0.5 ? -2 : (x[2] <= 0.3 ? 3 : 4)
    rng = Random.MersenneTwister(1234)
    features = rand(rng, 100, 2)
    labels = truth.(Vector.(eachrow(features)))
    ml_model = DecisionTree.build_tree(labels, features)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, 0 <= x[1:2] <= 1)
    y, formulation = MathOptAI.add_predictor(model, ml_model, x)
    @constraint(model, c_rhs, x .== 0.0)
    @objective(model, Min, sum(y))
    for _ in 1:10
        xi = rand(rng, 2)
        if minimum(abs.(xi .- [0.5, 0.3])) < 1e-2
            continue  # Skip points near kink
        end
        set_normalized_rhs.(c_rhs, xi)
        optimize!(model)
        @test ≈(value(only(y)), truth(xi); atol = 1e-6)
    end
    return
end

function test_DecisionTree()
    truth(x::Vector) = x[1] <= 0.5 ? -2 : (x[2] <= 0.3 ? 3 : 4)
    rng = Random.MersenneTwister(1234)
    features = rand(rng, 10, 2)
    labels = truth.(Vector.(eachrow(features)))
    ml_model = DecisionTree.DecisionTreeClassifier(; max_depth = 3)
    DecisionTree.fit!(ml_model, features, labels)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, 0 <= x[1:2] <= 1)
    y, formulation = MathOptAI.add_predictor(model, ml_model, x)
    @constraint(model, c_rhs, x .== 0.0)
    @objective(model, Min, sum(y))
    for _ in 1:10
        xi = rand(rng, 2)
        if minimum(abs.(xi .- [0.5, 0.3])) < 1e-2
            continue  # Skip points near kink
        end
        set_normalized_rhs.(c_rhs, xi)
        optimize!(model)
        @test ≈(value(only(y)), truth(xi); atol = 1e-6)
    end
    return
end

end  # module

TestDecisionTreeExt.runtests()
