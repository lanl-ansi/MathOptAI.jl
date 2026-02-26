# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module TestPredictors

using JuMP
using Test

import Distributions
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
    @test MathOptAI.output_size(f, (2,)) == (1,)
    y, formulation = MathOptAI.add_predictor(model, f, x)
    cons = all_constraints(model; include_variable_in_set_constraints = false)
    obj = constraint_object(only(cons))
    @test obj.set == MOI.EqualTo(0.0)
    @test isequal_canonical(obj.func, 2.0 * x[1] + 3.0 * x[2] - y[1])
    return
end

function test_Affine_constructors()
    # Affine(::Matrix)
    f = MathOptAI.Affine([1.0 2.0; 3.0 4.0])
    g = MathOptAI.Affine([1.0 2.0; 3.0 4.0], [0.0, 0.0])
    @test f.A == g.A && f.b == g.b
    # Affine(::Vector)
    f = MathOptAI.Affine([1.0, 2.0])
    g = MathOptAI.Affine([1.0 2.0], [0.0])
    @test f.A == g.A && f.b == g.b
    # Affine(::AbstractMatrix, ::AbstractVector)
    f = MathOptAI.Affine([1 2; 3 4], 5.0:6.0)
    g = MathOptAI.Affine([1.0 2.0; 3.0 4.0], [5.0, 6.0])
    @test f.A == g.A && f.b == g.b
    return
end

function test_Affine_affine()
    model = Model()
    @variable(model, x[1:2])
    f = MathOptAI.Affine([2.0, 3.0])
    y, formulation = MathOptAI.add_predictor(model, f, 2.0 .* x)
    cons = all_constraints(model; include_variable_in_set_constraints = false)
    obj = constraint_object(only(cons))
    @test obj.set == MOI.EqualTo(0.0)
    @test isequal_canonical(obj.func, 4.0 * x[1] + 6.0 * x[2] - y[1])
    return
end

function test_Affine_bounds()
    model = Model()
    @variable(model, x[1:2])
    fix(x[1], 2.0)
    set_binary(x[2])
    f = MathOptAI.Affine([2.0, 3.0])
    y, _ = MathOptAI.add_predictor(model, f, x)
    @test lower_bound.(y) == [4.0]  # 2 * (x[1]=2) + 3 * (x[2]=0)
    @test upper_bound.(y) == [7.0]  # 2 * (x[1]=2) + 3 * (x[2]=1)
    return
end

function test_ReducedSpace_Affine()
    model = Model()
    @variable(model, x[1:2])
    predictor = MathOptAI.ReducedSpace(MathOptAI.Affine([2.0, 3.0]))
    y, formulation = MathOptAI.add_predictor(model, predictor, x)
    cons = all_constraints(model; include_variable_in_set_constraints = false)
    @test isempty(cons)
    @test isequal_canonical(y, [2x[1] + 3x[2]])
    return
end

function test_BinaryDecisionTree()
    rhs = MathOptAI.BinaryDecisionTree{Float64,Int}(1, 1.0, 0, 1)
    f = MathOptAI.BinaryDecisionTree{Float64,Int}(1, 0.0, -1, rhs)
    @test MathOptAI.output_size(f, (1,)) == (1,)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, -3 <= x <= 5)
    y, formulation = MathOptAI.add_predictor(model, f, [x])
    @constraint(model, c_rhs, x == 0.0)
    for (xi, yi) in (-0.4 => -1, -0.3 => -1, 0.4 => 0, 1.3 => 1)
        set_normalized_rhs(c_rhs, xi)
        optimize!(model)
        @test ≈(value(only(y)), yi; atol = 1e-6)
    end
    return
end

function test_Quantile()
    model = Model(Ipopt.Optimizer)
    @variable(model, 1 <= x <= 2)
    predictor = MathOptAI.Quantile([0.1, 0.9]) do x
        return Distributions.Normal(x, 3 - x)
    end
    y, formulation = MathOptAI.add_predictor(model, predictor, [x])
    @objective(model, Min, y[2] - y[1])
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test ≈(value(x), 2; atol = 1e-4)
    d = Distributions.Normal(value(x), 3 - value(x))
    y_target = map(q -> Distributions.quantile(d, q), [0.1, 0.9])
    @test ≈(value.(y), y_target; atol = 1e-4)
    return
end

function test_RandomForest()
    #     x <= 0
    #      /  \
    #   -1     x <= 1
    #           /  \
    #         0      1
    rhs = MathOptAI.BinaryDecisionTree{Float64,Int}(1, 1.0, 0, 1)
    tree_1 = MathOptAI.BinaryDecisionTree{Float64,Int}(1, 0.0, -1, rhs)
    #        x <= 0.9
    #        /      \
    #    x <= -0.1    1
    #     /  \
    #  -1     0
    lhs = MathOptAI.BinaryDecisionTree{Float64,Int}(1, -0.1, -1, 0)
    tree_2 = MathOptAI.BinaryDecisionTree{Float64,Int}(1, 0.9, lhs, 1)
    predictor = MathOptAI.AffineCombination([tree_1, tree_2], [0.5, 0.5], [0.0])
    @test MathOptAI.output_size(predictor, (1,)) == (1,)
    @test sprint(show, predictor) == """
    AffineCombination
    ├ 0.5 * BinaryDecisionTree{Float64,Int64} [leaves=3, depth=2]
    ├ 0.5 * BinaryDecisionTree{Float64,Int64} [leaves=3, depth=2]
    └ 1.0 * [0.0]"""
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, -3 <= x <= 5)
    y, formulation = MathOptAI.add_predictor(model, predictor, [x])
    @constraint(model, c_rhs, x == 0.0)
    for (xi, yi) in (-0.4 => -1, -0.05 => -0.5, 0.4 => 0, 0.95 => 0.5, 1.3 => 1)
        set_normalized_rhs(c_rhs, xi)
        optimize!(model)
        @test ≈(value(only(y)), yi; atol = 1e-6)
    end
    return
end

function test_ReLU_direct()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    f = MathOptAI.ReLU()
    @test MathOptAI.output_size(f, (10,)) == (10,)
    y, formulation = MathOptAI.add_predictor(model, f, x)
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

function test_ReLU_bounds()
    values = [-2, 0, 2]
    for f in (
        MathOptAI.ReLU(),
        MathOptAI.ReLUBigM(100.0),
        MathOptAI.ReLUQuadratic(),
        MathOptAI.ReLUSOS1(),
    )
        for lb in values, ub in values
            if lb > ub
                continue
            end
            model = Model()
            @variable(model, lb <= x <= ub)
            y, _ = MathOptAI.add_predictor(model, f, [x])
            @test lower_bound.(y) == [max(0.0, lb)]
            @test upper_bound.(y) == [max(0.0, ub)]
        end
    end
    return
end

function test_ReducedSpace_ReLU_direct()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    f = MathOptAI.ReducedSpace(MathOptAI.ReLU())
    y, formulation = MathOptAI.add_predictor(model, f, x)
    @test length(y) == 2
    @test num_variables(model) == 2
    @test num_constraints(model, NonlinearExpr, MOI.EqualTo{Float64}) == 0
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
    @test MathOptAI.output_size(f, (10,)) == (10,)
    y, formulation = MathOptAI.add_predictor(model, f, x)
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
    @test MathOptAI.output_size(f, (10,)) == (10,)
    y, formulation = MathOptAI.add_predictor(model, f, x)
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
    y, formulation = MathOptAI.add_predictor(model, MathOptAI.ReLUSOS1(), x)
    err = if isdefined(MOI.Bridges, :BridgeRequiresFiniteDomainError)
        MOI.Bridges.BridgeRequiresFiniteDomainError
    else
        ErrorException(
            "Unable to use SOS1ToMILPBridge because element 1 in the function has a non-finite domain: MOI.VariableIndex(3)",
        )
    end
    @test_throws err optimize!(model)
    return
end

function test_ReLU_Quadratic()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    f = MathOptAI.ReLUQuadratic()
    @test MathOptAI.output_size(f, (10,)) == (10,)
    @test f.relaxation_parameter === nothing
    y, formulation = MathOptAI.add_predictor(model, f, x)
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

function test_ReLU_Quadratic_relaxed()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    f = MathOptAI.ReLUQuadratic(; relaxation_parameter = 1e-4)
    y, formulation = MathOptAI.add_predictor(model, f, x)
    # Maximize sum of all variables to exercise the ReLU relaxation
    @objective(model, Max, sum(formulation.variables))
    @test length(y) == 2
    @test num_variables(model) == 6
    @test num_constraints(model, AffExpr, MOI.EqualTo{Float64}) == 2
    @test num_constraints(model, QuadExpr, MOI.LessThan{Float64}) == 2
    fix.(x, [-1, 2])
    optimize!(model)
    @assert is_solved_and_feasible(model)
    # We do not satisfy equality to a tight tolerance
    @test !isapprox(value.(y), [0.0, 2.0]; atol = 1e-6)
    # But we satisfy equality to a loose tolerance
    @test isapprox(value.(y), [0.0, 2.0]; atol = 1e-2)
    return
end

function test_Sigmoid()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    f = MathOptAI.Sigmoid()
    @test MathOptAI.output_size(f, (10,)) == (10,)
    y, formulation = MathOptAI.add_predictor(model, f, x)
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

function test_Sigmoid_bounds()
    f(x) = 1 / (1 + exp(-x))
    values = [-Inf, -2, 0, 2, Inf]
    for lb in values, ub in values
        if lb == Inf || ub == -Inf || lb > ub
            continue
        end
        model = Model()
        @variable(model, lb <= x <= ub)
        y, _ = MathOptAI.add_predictor(model, MathOptAI.Sigmoid(), [x])
        @test lower_bound(y[1]) == f(lb)
        @test upper_bound(y[1]) == f(ub)
    end
    return
end

function test_ReducedSpace_Sigmoid()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    predictor = MathOptAI.ReducedSpace(MathOptAI.Sigmoid())
    y, formulation = MathOptAI.add_predictor(model, predictor, x)
    @test length(y) == 2
    @test num_variables(model) == 2
    @test num_constraints(model, NonlinearExpr, MOI.EqualTo{Float64}) == 0
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
    y, formulation = MathOptAI.add_predictor(model, MathOptAI.SoftMax(), x)
    @test MathOptAI.output_size(MathOptAI.SoftMax(), (10,)) == (10,)
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

function test_ReducedSpace_SoftMax()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    predictor = MathOptAI.ReducedSpace(MathOptAI.SoftMax())
    y, formulation = MathOptAI.add_predictor(model, predictor, x)
    @test length(y) == 2
    @test num_variables(model) == 3
    @test num_constraints(model, NonlinearExpr, MOI.EqualTo{Float64}) == 1
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
    y, formulation = MathOptAI.add_predictor(model, MathOptAI.SoftPlus(), x)
    @test MathOptAI.output_size(MathOptAI.SoftPlus(), (10,)) == (10,)
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

function test_SoftPlus_bounds()
    f(x, beta) = log(1 + exp(beta * x)) / beta
    values = [-Inf, -2, 0, 2, Inf]
    for beta in [1.0, 1.5, 2.0], lb in values, ub in values
        if lb == Inf || ub == -Inf || lb > ub
            continue
        end
        model = Model()
        @variable(model, lb <= x <= ub)
        y, _ = MathOptAI.add_predictor(model, MathOptAI.SoftPlus(; beta), [x])
        @test lower_bound(y[1]) == f(lb, beta)
        if isfinite(ub)
            @test upper_bound(y[1]) == f(ub, beta)
        else
            @test !has_upper_bound(y[1])
        end
    end
    return
end

function test_ReducedSpace_SoftPlus()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    predictor = MathOptAI.ReducedSpace(MathOptAI.SoftPlus(; beta = 1.1))
    y, formulation = MathOptAI.add_predictor(model, predictor, x)
    @test length(y) == 2
    @test num_variables(model) == 2
    @test num_constraints(model, NonlinearExpr, MOI.EqualTo{Float64}) == 0
    @objective(model, Min, sum(y))
    X = [-1.0, 2.0]
    fix.(x, X)
    optimize!(model)
    @assert is_solved_and_feasible(model)
    @test value.(y) ≈ log.(1 .+ exp.(X .* 1.1)) ./ 1.1
    return
end

function test_Tanh()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    y, formulation = MathOptAI.add_predictor(model, MathOptAI.Tanh(), x)
    @test MathOptAI.output_size(MathOptAI.Tanh(), (10,)) == (10,)
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

function test_Tanh_bounds()
    values = [-Inf, -2, 0, 2, Inf]
    for lb in values, ub in values
        if lb == Inf || ub == -Inf || lb > ub
            continue
        end
        model = Model()
        @variable(model, lb <= x <= ub)
        y, _ = MathOptAI.add_predictor(model, MathOptAI.Tanh(), [x])
        @test lower_bound.(y) == [tanh(lb)]
        @test upper_bound.(y) == [tanh(ub)]
    end
    return
end

function test_ReducedSpace_Tanh()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    predictor = MathOptAI.ReducedSpace(MathOptAI.Tanh())
    y, formulation = MathOptAI.add_predictor(model, predictor, x)
    @test length(y) == 2
    @test num_variables(model) == 2
    @test num_constraints(model, NonlinearExpr, MOI.EqualTo{Float64}) == 0
    @objective(model, Min, sum(y))
    X = [-1.0, 2.0]
    fix.(x, X)
    optimize!(model)
    @assert is_solved_and_feasible(model)
    @test value.(y) ≈ tanh.(X)
    return
end

function test_ReducedSpace_ReducedSpace()
    predictor = MathOptAI.ReducedSpace(MathOptAI.Tanh())
    @test MathOptAI.ReducedSpace(predictor) === predictor
    return
end

function test_Scale()
    model = Model()
    @variable(model, x[1:2])
    f = MathOptAI.Scale([2.0, 3.0], [4.0, 5.0])
    @test MathOptAI.output_size(MathOptAI.Tanh(), (2,)) == (2,)
    @test sprint(show, f) == "Scale(scale, bias)"
    y, formulation = MathOptAI.add_predictor(model, f, x)
    cons = all_constraints(model; include_variable_in_set_constraints = false)
    @test length(cons) == 2
    objs = constraint_object.(cons)
    @test objs[1].set == MOI.EqualTo(-4.0)
    @test objs[2].set == MOI.EqualTo(-5.0)
    @test isequal_canonical(objs[1].func, 2.0 * x[1] - y[1])
    @test isequal_canonical(objs[2].func, 3.0 * x[2] - y[2])
    y, formulation =
        MathOptAI.add_predictor(model, MathOptAI.ReducedSpace(f), x)
    cons = all_constraints(model; include_variable_in_set_constraints = false)
    @test length(cons) == 2
    @test isequal_canonical(y, [2.0 * x[1] + 4.0, 3.0 * x[2] + 5.0])
    return
end

function test_Scale_constructor()
    # Scale(::AbstractVector, ::AbstractVector)
    f = MathOptAI.Scale([1 // 2, 2 // 5], 3.0:4.0)
    g = MathOptAI.Scale([0.5, 0.4], [3.0, 4.0])
    @test f.scale == g.scale && f.bias == g.bias
    return
end

function test_fallback_bound_methods()
    l, u = MathOptAI.get_variable_bounds("x")
    @test ismissing(l) && ismissing(u)
    optional = true
    @test MathOptAI.set_variable_bounds(Any[], "x", l, u; optional) === nothing
    @test MathOptAI.set_variable_bounds(Any[], "x", 0, 1; optional) === nothing
    optional = false
    @test MathOptAI.set_variable_bounds(Any[], "x", l, u; optional) === nothing
    @test_throws(
        ErrorException("You must implement this method."),
        MathOptAI.set_variable_bounds(Any[], "x", 0, 1; optional)
    )
    return
end

function test_Affine_DimensionMismatch()
    @test_throws DimensionMismatch MathOptAI.Affine([1 2; 3 4], [5, 6, 7])
    model = Model()
    @variable(model, x[1:2])
    f = MathOptAI.Affine([1 2 3; 4 5 6], [7, 8])
    @test_throws DimensionMismatch MathOptAI.add_predictor(model, f, x)
    return
end

function test_Scale_DimensionMismatch()
    @test_throws DimensionMismatch MathOptAI.Scale([1, 2], [5, 6, 7])
    model = Model()
    @variable(model, x[1:2])
    f = MathOptAI.Scale([1, 2, 3], [4, 5, 6])
    @test_throws DimensionMismatch MathOptAI.add_predictor(model, f, x)
    return
end

function test_GELU()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    y, formulation = MathOptAI.add_predictor(model, MathOptAI.GELU(), x)
    @test MathOptAI.output_size(MathOptAI.GELU(), (10,)) == (10,)
    @test length(y) == 2
    @test num_variables(model) == 4
    @test num_constraints(model, NonlinearExpr, MOI.EqualTo{Float64}) == 2
    @objective(model, Min, sum(y))
    X = [-1.0, 2.0]
    fix.(x, X)
    optimize!(model)
    @assert is_solved_and_feasible(model)
    @test value.(y) ≈ MathOptAI.GELU().(X)
    return
end

function test_GELU_bounds()
    values = [-Inf, -1, 0, 2, Inf]
    for lb in values, ub in values
        if lb == Inf || ub == -Inf || lb > ub
            continue
        end
        model = Model()
        @variable(model, lb <= x <= ub)
        y, _ = MathOptAI.add_predictor(model, MathOptAI.GELU(), [x])
        if lb >= 0.0
            @test lower_bound.(y) == [MathOptAI.GELU()(lb)]
        else
            @test lower_bound.(y) == [-0.17]
        end
        if ub == Inf
            @test !any(has_upper_bound.(y))
        elseif ub >= 0.0
            @test upper_bound.(y) == [MathOptAI.GELU()(ub)]
        else
            @test upper_bound.(y) == [0.0]
        end
    end
    return
end

function test_ReducedSpace_GELU()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    predictor = MathOptAI.ReducedSpace(MathOptAI.GELU())
    y, formulation = MathOptAI.add_predictor(model, predictor, x)
    @test length(y) == 2
    @test num_variables(model) == 2
    @test num_constraints(model, NonlinearExpr, MOI.EqualTo{Float64}) == 0
    @objective(model, Min, sum(y))
    X = [-1.0, 2.0]
    fix.(x, X)
    optimize!(model)
    @assert is_solved_and_feasible(model)
    @test value.(y) ≈ MathOptAI.GELU().(X)
    return
end

function test_start_values()
    A = [sin(i + j) for i in 1:3, j in 1:3]
    b = [cos(i) for i in 1:3]
    _compare(::Nothing, ::Nothing) = true
    _compare(x, y) = ≈(x, y)
    _compare(x::AbstractVector, y::AbstractVector) = all(_compare.(x, y))
    layers = Any[
        MathOptAI.Affine(A, b)=>(x->A*x .+ b),
        MathOptAI.GELU()=>(x->@.(0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))))),
        MathOptAI.ReLU()=>(x->max.(0, x)),
        MathOptAI.Scale(b, -b)=>(x->b .* x .- b),
        MathOptAI.Sigmoid()=>(x->1 ./ (1 .+ exp.(-x))),
        MathOptAI.SoftMax()=>(x->exp.(x) ./ sum(exp.(x))),
        MathOptAI.SoftPlus(; beta = 2.0)=>(x->0.5 .* log.(1 .+ exp.(2 .* x))),
        MathOptAI.Tanh()=>(x->tanh.(x)),
    ]
    for (layer, fn) in layers
        model = Model()
        x0 = [-0.5, 0.0, 1.1]
        @variable(model, x[i in 1:3], start = x0[i])
        y, _ = MathOptAI.add_predictor(model, layer, x)
        @test _compare(start_value.(y), fn(x0))
        y, _ = MathOptAI.add_predictor(model, layer, 2.0 .* x .+ 3.0)
        @test _compare(start_value.(y), fn(2.0 .* x0 .+ 3.0))
    end
    for (layer, fn) in layers
        model = Model()
        x0 = [0.5, nothing, -1.1]
        @variable(model, x[i in 1:3], start = x0[i])
        y, _ = MathOptAI.add_predictor(model, layer, x)
        xm = something.(x0, missing)
        @test _compare(start_value.(y), coalesce.(fn(xm), nothing))
        y, _ = MathOptAI.add_predictor(model, layer, 2.0 .* x .+ 3.0)
        xm = something.(2.0 .* xm .+ 3.0, missing)
        @test _compare(start_value.(y), coalesce.(fn(xm), nothing))
    end
    return
end

function test_start_values_constant()
    model = Model()
    @variable(model, x, start = 2)
    y, _ = MathOptAI.add_predictor(model, MathOptAI.Sigmoid(), Any[x, 1.0])
    @test start_value.(y) ≈ 1 ./ (1 .+ exp.([-2, -1]))
    return
end

function test_start_values_missing()
    # A fallback for types that we don't know about, like InfiniteOpt extensions
    @test MathOptAI.get_variable_start(:a) === missing
    return
end

function test_LeakyReLU()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    f = MathOptAI.LeakyReLU(; negative_slope = 0.01)
    @test MathOptAI.output_size(f, (10,)) == (10,)
    y, formulation = MathOptAI.add_predictor(model, f, x)
    @test length(y) == 2
    @test num_variables(model) == 6
    @test num_constraints(model, NonlinearExpr, MOI.EqualTo{Float64}) == 2
    @test num_constraints(model, AffExpr, MOI.EqualTo{Float64}) == 2
    @objective(model, Min, sum(y))
    fix.(x, [-1, 2])
    optimize!(model)
    @assert is_solved_and_feasible(model)
    @test value.(y) ≈ [-0.01, 2.0]
    return
end

function test_LeakyReLU_reduced_space()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    f = MathOptAI.LeakyReLU(; negative_slope = 0.2)
    y, formulation = MathOptAI.add_predictor(model, f, x; reduced_space = true)
    @test length(y) == 2
    @test num_variables(model) == 2
    @test num_constraints(model, NonlinearExpr, MOI.EqualTo{Float64}) == 0
    @test num_constraints(model, AffExpr, MOI.EqualTo{Float64}) == 0
    @objective(model, Min, sum(y))
    fix.(x, [-1, 2])
    optimize!(model)
    @assert is_solved_and_feasible(model)
    @test value.(y) ≈ [-0.2, 2.0]
    return
end

function test_LeakyReLU_BigM()
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    f = MathOptAI.LeakyReLU(;
        negative_slope = 0.123,
        relu = MathOptAI.ReLUBigM(100.0),
    )
    y, formulation = MathOptAI.add_predictor(model, f, x)
    @test length(y) == 2
    @test num_variables(model) == 8
    @test num_constraints(model, AffExpr, MOI.EqualTo{Float64}) == 2
    @test num_constraints(model, AffExpr, MOI.GreaterThan{Float64}) == 2
    @test num_constraints(model, AffExpr, MOI.LessThan{Float64}) == 4
    @objective(model, Min, sum(y))
    fix.(x, [-1, 2])
    optimize!(model)
    @assert is_solved_and_feasible(model)
    @test value.(y) ≈ [-0.123, 2.0]
    return
end

function test_AvgPool2d()
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[h in 1:2, w in 1:4] == w + 4 * (h - 1))
    predictor = MathOptAI.AvgPool2d((2, 2); input_size = (2, 4, 1))
    @test MathOptAI.output_size(predictor, (2, 4)) == (1, 2, 1)
    @test MathOptAI.output_size(predictor, (2, 4, 1)) == (1, 2, 1)
    @test MathOptAI.output_size(predictor, nothing) === nothing
    y, formulation = MathOptAI.add_predictor(model, predictor, vec(x))
    @test num_constraints(model, AffExpr, MOI.EqualTo{Float64}) == 2
    optimize!(model)
    @assert is_solved_and_feasible(model)
    @test value.(y) ≈ [3.5, 5.5]
    return
end

function test_AvgPool2d_reduced_space()
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[h in 1:2, w in 1:4] == w + 4 * (h - 1))
    predictor = MathOptAI.AvgPool2d((2, 2); input_size = (2, 4, 1))
    y, formulation =
        MathOptAI.add_predictor(model, predictor, vec(x); reduced_space = true)
    @test num_constraints(model, AffExpr, MOI.EqualTo{Float64}) == 0
    optimize!(model)
    @assert is_solved_and_feasible(model)
    @test value.(y) ≈ [3.5, 5.5]
    return
end

function test_Conv2d()
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[h in 1:2, w in 1:4] == w + 4 * (h - 1))
    weight = reshape(sin.(1:8), 2, 2, 1, 2)
    bias = [1.0, 2.0]
    predictor = MathOptAI.Conv2d(weight, bias; input_size = (2, 4, 1))
    @test MathOptAI.output_size(predictor, (2, 4)) == (1, 3, 2)
    @test MathOptAI.output_size(predictor, (2, 4, 1)) == (1, 3, 2)
    y, formulation = MathOptAI.add_predictor(model, predictor, vec(x))
    @test num_constraints(model, AffExpr, MOI.EqualTo{Float64}) == 6
    optimize!(model)
    @assert is_solved_and_feasible(model)
    z = value(x)
    y_star = [
        sum(z[:, i:(i+1)] .* weight[2:-1:1, 2:-1:1, 1, j]) + bias[j] for
        j in 1:2 for i in 1:3
    ]
    @test value.(y) ≈ y_star
    return
end

function test_Conv2d_reduced_space()
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[h in 1:2, w in 1:4] == w + 4 * (h - 1))
    weight = reshape(sin.(1:8), 2, 2, 1, 2)
    bias = [1.0, 2.0]
    predictor = MathOptAI.Conv2d(weight, bias; input_size = (2, 4, 1))
    y, formulation =
        MathOptAI.add_predictor(model, predictor, vec(x); reduced_space = true)
    @test num_constraints(model, AffExpr, MOI.EqualTo{Float64}) == 0
    optimize!(model)
    @assert is_solved_and_feasible(model)
    z = value(x)
    y_star = [
        sum(z[:, i:(i+1)] .* weight[2:-1:1, 2:-1:1, 1, j]) + bias[j] for
        j in 1:2 for i in 1:3
    ]
    @test value.(y) ≈ y_star
    return
end

function test_MaxPool2d()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[h in 1:2, w in 1:4] == w + 4 * (h - 1))
    predictor = MathOptAI.MaxPool2d((2, 2); input_size = (2, 4, 1))
    @test MathOptAI.output_size(predictor, (2, 4)) == (1, 2, 1)
    @test MathOptAI.output_size(predictor, (2, 4, 1)) == (1, 2, 1)
    y, formulation = MathOptAI.add_predictor(model, predictor, vec(x))
    @test num_constraints(model, NonlinearExpr, MOI.EqualTo{Float64}) == 2
    optimize!(model)
    @assert is_solved_and_feasible(model)
    @test value.(y) ≈ [6, 8]
    return
end

function test_MaxPool2d_reduced_space()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[h in 1:2, w in 1:4] == w + 4 * (h - 1))
    predictor = MathOptAI.MaxPool2d((2, 2); input_size = (2, 4, 1))
    y, formulation =
        MathOptAI.add_predictor(model, predictor, vec(x); reduced_space = true)
    @test num_constraints(model, NonlinearExpr, MOI.EqualTo{Float64}) == 0
    optimize!(model)
    @assert is_solved_and_feasible(model)
    @test value.(y) ≈ [6, 8]
    return
end

function test_MaxPool2dBigM()
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[h in 1:2, w in 1:4] == w + 4 * (h - 1))
    predictor =
        MathOptAI.MaxPool2dBigM((2, 2); input_size = (2, 4, 1), M = 100.0)
    @test MathOptAI.output_size(predictor, (2, 4)) == (1, 2, 1)
    @test MathOptAI.output_size(predictor, (2, 4, 1)) == (1, 2, 1)
    y, formulation = MathOptAI.add_predictor(model, predictor, vec(x))
    optimize!(model)
    @assert is_solved_and_feasible(model)
    @test value.(y) ≈ [6, 8]
    return
end

function test_pipeline()
    input_size = (4, 4, 1)
    layers = [MathOptAI.MaxPool2d((2, 2); input_size), MathOptAI.ReLU()]
    p = MathOptAI.Pipeline(layers)
    @test MathOptAI.output_size(p, nothing) === nothing
    @test MathOptAI.output_size(p, (4, 4)) === (2, 2, 1)
    @test MathOptAI.output_size(p, (4, 4, 1)) === (2, 2, 1)
    return
end

function test_GCNConv()
    predictor = MathOptAI.GCNConv(;
        weights = [1.0 2.0; 3.0 4.0; 5.0 6.0],
        bias = [7.0, 8.0],
        edge_index = [1 => 2, 2 => 1, 2 => 3, 3 => 2, 3 => 4, 4 => 3],
    )
    @test MathOptAI.output_size(predictor, nothing) === (4, 2)
    @test MathOptAI.output_size(predictor, (4, 3)) === (4, 2)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[i in 1:4, j in 1:3] == i + j)
    y, _ = MathOptAI.add_predictor(model, predictor, vec(x))
    optimize!(model)
    assert_is_solved_and_feasible(model)
    @test round.(Int, value(y)) == [39, 49, 60, 56, 49, 63, 78, 72]
    return
end

function test_GCNConv_reduced_space()
    predictor = MathOptAI.GCNConv(;
        weights = [1.0 2.0; 3.0 4.0; 5.0 6.0],
        bias = [7.0, 8.0],
        edge_index = [1 => 2, 2 => 1, 2 => 3, 3 => 2, 3 => 4, 4 => 3],
    )
    @test MathOptAI.output_size(predictor, nothing) === (4, 2)
    @test MathOptAI.output_size(predictor, (4, 3)) === (4, 2)
    model = Model()
    @variable(model, x[i in 1:4, j in 1:3] == i + j)
    y, _ =
        MathOptAI.add_predictor(model, predictor, vec(x); reduced_space = true)
    @test round.(Int, value(fix_value, y)) == [39, 49, 60, 56, 49, 63, 78, 72]
    return
end

function test_TAGConv()
    predictor = MathOptAI.TAGConv(;
        weights = [[1.0 2.0; 3.0 4.0; 5.0 6.0], [1.0 2.0; 3.0 4.0; 5.0 6.0]],
        bias = [7.0, 8.0],
        edge_index = [1 => 2, 2 => 1, 2 => 3, 3 => 2, 3 => 4, 4 => 3],
    )
    @test MathOptAI.output_size(predictor, nothing) === (4, 2)
    @test MathOptAI.output_size(predictor, (4, 3)) === (4, 2)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[i in 1:4, j in 1:3] == i + j)
    y, _ = MathOptAI.add_predictor(model, predictor, vec(x))
    optimize!(model)
    assert_is_solved_and_feasible(model)
    @test round.(Int, value(y)) == [66, 93, 117, 100, 85, 120, 152, 129]
    return
end

function test_TAGConv_reduced_space()
    predictor = MathOptAI.TAGConv(;
        weights = [[1.0 2.0; 3.0 4.0; 5.0 6.0], [1.0 2.0; 3.0 4.0; 5.0 6.0]],
        bias = [7.0, 8.0],
        edge_index = [1 => 2, 2 => 1, 2 => 3, 3 => 2, 3 => 4, 4 => 3],
    )
    @test MathOptAI.output_size(predictor, nothing) === (4, 2)
    @test MathOptAI.output_size(predictor, (4, 3)) === (4, 2)
    model = Model()
    @variable(model, x[i in 1:4, j in 1:3] == i + j)
    y, _ =
        MathOptAI.add_predictor(model, predictor, vec(x); reduced_space = true)
    @test round.(Int, value(fix_value, y)) ==
          [66, 93, 117, 100, 85, 120, 152, 129]
    return
end

function test_add_predictor_kwarg_err()
    model = Model()
    @variable(model, x[1:2, 1:2])
    predictor = MathOptAI.ReLU()
    @test_throws(
        ErrorException(
            """
            The predictor $(typeof(predictor)) does not support `::Array` inputs`.
            You must first vectorize the input by calling `vec(x)`.
            """,
        ),
        MathOptAI.add_predictor(model, predictor, x),
    )
    @test_throws(
        MethodError,
        MathOptAI.add_predictor(model, predictor, vec(x); input_size = (2, 2)),
    )
    return
end

end  # module

TestPredictors.runtests()
