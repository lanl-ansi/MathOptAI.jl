# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module PredictorTests

using JuMP
using Test

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

function test_Affine()
    model = Model()
    @variable(model, x[1:2])
    f = MathOptAI.Affine([2.0, 3.0])
    y = MathOptAI.add_predictor(model, f, x)
    cons = all_constraints(model; include_variable_in_set_constraints = false)
    obj = constraint_object(only(cons))
    @test obj.set == MOI.EqualTo(0.0)
    @test isequal_canonical(obj.func, 2.0 * x[1] + 3.0 * x[2] - y[1])
    return
end

function test_Affine_affine()
    model = Model()
    @variable(model, x[1:2])
    f = MathOptAI.Affine([2.0, 3.0])
    y = MathOptAI.add_predictor(model, f, 2.0 .* x)
    cons = all_constraints(model; include_variable_in_set_constraints = false)
    obj = constraint_object(only(cons))
    @test obj.set == MOI.EqualTo(0.0)
    @test isequal_canonical(obj.func, 4.0 * x[1] + 6.0 * x[2] - y[1])
    return
end

function test_ReLU_direct()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    f = MathOptAI.ReLU()
    y = MathOptAI.add_predictor(model, f, x)
    @test length(y) == 2
    @test num_variables(model) == 4
    @test num_constraints(model, NonlinearExpr, MOI.EqualTo{Float64}) == 2
    @objective(model, Min, sum(y))
    fix.(x, [-1, 2])
    optimize!(model)
    @assert is_solved_and_feasible(model)
    @test value.(y) ≈ [0.0, 2.0]
    return
end

function test_ReLU_BigM()
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    f = MathOptAI.ReLUBigM(100.0)
    y = MathOptAI.add_predictor(model, f, x)
    @test length(y) == 2
    @test num_variables(model) == 6
    @test num_constraints(model, AffExpr, MOI.LessThan{Float64}) == 4
    @test num_constraints(model, AffExpr, MOI.GreaterThan{Float64}) == 2
    @objective(model, Min, sum(y))
    fix.(x, [-1, 2])
    optimize!(model)
    @assert is_solved_and_feasible(model)
    @test value.(y) ≈ [0.0, 2.0]
    return
end

function test_ReLU_SOS1()
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, -2 <= x[1:2] <= 2)
    f = MathOptAI.ReLUSOS1()
    y = MathOptAI.add_predictor(model, f, x)
    @test length(y) == 2
    @test num_variables(model) == 6
    @test num_constraints(model, Vector{VariableRef}, MOI.SOS1{Float64}) == 2
    @objective(model, Min, sum(y))
    @constraint(model, x .>= [-1, 2])
    optimize!(model)
    @assert is_solved_and_feasible(model)
    @test value.(y) ≈ [0.0, 2.0]
    return
end

function test_ReLU_SOS1_no_bounds()
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    y = MathOptAI.add_predictor(model, MathOptAI.ReLUSOS1(), x)
    @test_throws(
        ErrorException(
            "Unable to use SOS1ToMILPBridge because element 1 in the function has a non-finite domain: MOI.VariableIndex(1)",
        ),
        optimize!(model),
    )
    return
end

function test_ReLU_Quadratic()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    f = MathOptAI.ReLUQuadratic()
    y = MathOptAI.add_predictor(model, f, x)
    @test length(y) == 2
    @test num_variables(model) == 6
    @test num_constraints(model, QuadExpr, MOI.EqualTo{Float64}) == 2
    @objective(model, Min, sum(y))
    fix.(x, [-1, 2])
    optimize!(model)
    @assert is_solved_and_feasible(model)
    @test value.(y) ≈ [0.0, 2.0]
    return
end

function test_Sigmoid()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    y = MathOptAI.add_predictor(model, MathOptAI.Sigmoid(), x)
    @test length(y) == 2
    @test num_variables(model) == 4
    @test num_constraints(model, NonlinearExpr, MOI.EqualTo{Float64}) == 2
    @objective(model, Min, sum(y))
    X = [-1.0, 2.0]
    fix.(x, X)
    optimize!(model)
    @assert is_solved_and_feasible(model)
    @test value.(y) ≈ 1 ./ (1 .+ exp.(-X))
    return
end

function test_SoftMax()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    y = MathOptAI.add_predictor(model, MathOptAI.SoftMax(), x)
    @test length(y) == 2
    @test num_variables(model) == 5
    @test num_constraints(model, NonlinearExpr, MOI.EqualTo{Float64}) == 3
    @objective(model, Min, sum(y))
    X = [-1.0, 2.0]
    fix.(x, X)
    optimize!(model)
    @assert is_solved_and_feasible(model)
    @test value.(y) ≈ exp.(X) ./ sum(exp.(X))
    return
end

function test_SoftPlus()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    y = MathOptAI.add_predictor(model, MathOptAI.SoftPlus(), x)
    @test length(y) == 2
    @test num_variables(model) == 4
    @test num_constraints(model, NonlinearExpr, MOI.EqualTo{Float64}) == 2
    @objective(model, Min, sum(y))
    X = [-1.0, 2.0]
    fix.(x, X)
    optimize!(model)
    @assert is_solved_and_feasible(model)
    @test value.(y) ≈ log.(1 .+ exp.(X))
    return
end

function test_Tanh()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    y = MathOptAI.add_predictor(model, MathOptAI.Tanh(), x)
    @test length(y) == 2
    @test num_variables(model) == 4
    @test num_constraints(model, NonlinearExpr, MOI.EqualTo{Float64}) == 2
    @objective(model, Min, sum(y))
    X = [-1.0, 2.0]
    fix.(x, X)
    optimize!(model)
    @assert is_solved_and_feasible(model)
    @test value.(y) ≈ tanh.(X)
    return
end

end  # module

PredictorTests.runtests()
