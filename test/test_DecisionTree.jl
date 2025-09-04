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

function test_DecisionTreeClassifier()
    truth(x::Vector) = x[1] <= 0.5 ? -2 : (x[2] <= 0.3 ? 3 : 4)
    rng = Random.MersenneTwister(1234)
    features = rand(rng, 100, 2)
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

function test_DecisionTree_RandomForest()
    truth(x::Vector) = x[1] <= 0.5 ? -2 : (x[2] <= 0.3 ? 3 : 4)
    rng = Random.MersenneTwister(1234)
    features = rand(rng, 100, 2)
    labels = truth.(Vector.(eachrow(features)))
    ml_model = DecisionTree.build_forest(labels, features)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, 0 <= x[1:2] <= 1)
    y, formulation = MathOptAI.add_predictor(model, ml_model, x)
    @constraint(model, c_rhs, x .== 0.0)
    @objective(model, Min, sum(y))
    tree_y = filter!(mapreduce(f -> f.variables, vcat, formulation.layers)) do v
        return occursin("_value", name(v))
    end
    for _ in 1:10
        xi = rand(rng, 2)
        if minimum(abs.(xi .- [0.5, 0.3])) < 1e-2
            continue  # Skip points near kink
        end
        set_normalized_rhs.(c_rhs, xi)
        optimize!(model)
        tree_values = value.(tree_y)
        @test ≈(value(only(y)), sum(tree_values / length(tree_y)); atol = 1e-6)
    end
    return
end

function test_issue_195()
    predictor = DecisionTree.Root(
        DecisionTree.Node(
            1,
            0.0,
            DecisionTree.Leaf(2.0, [2.0]),
            DecisionTree.Leaf(0.0, [0.0]),
        ),
        1,
        [1.0],
    )
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, -1 <= x[1:1] <= 5)
    y, _ = MathOptAI.add_predictor(model, predictor, x)
    @objective(model, Max, only(y))
    optimize!(model)
    assert_is_solved_and_feasible(model)
    @test ≈(value.(x), [-1e-6]; atol = 1e-7)
    @test ≈(objective_value(model), 2.0; atol = 1e-6)
    @test ≈(DecisionTree.apply_tree(predictor, value.(x)), 2.0; atol = 1e-5)
    return
end

function test_issue_195_atol_0()
    predictor = DecisionTree.Root(
        DecisionTree.Node(
            1,
            0.0,
            DecisionTree.Leaf(2.0, [2.0]),
            DecisionTree.Leaf(0.0, [0.0]),
        ),
        1,
        [1.0],
    )
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, -1 <= x[1:1] <= 5)
    y, _ = MathOptAI.add_predictor(model, predictor, x; atol = 0.0)
    @objective(model, Max, only(y))
    optimize!(model)
    assert_is_solved_and_feasible(model)
    # Because tree is x <= 0 - atol ==> 2 : 0, HiGHS will choose x = 0, y = 2,
    # but the  tree will choose x = 0, y = 0.
    @test ≈(value.(x), [0.0]; atol = 1e-6)
    @test ≈(objective_value(model), 2.0; atol = 1e-6)
    @test ≈(DecisionTree.apply_tree(predictor, value.(x)), 0.0; atol = 1e-5)
    return
end

end  # module

TestDecisionTreeExt.runtests()
