# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module ReLUTests

using JuMP
using Test

import HiGHS
import Ipopt
import Omelette

is_test(x) = startswith(string(x), "test_")

function runtests()
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

function test_ReLU_BigM()
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    f = Omelette.ReLUBigM(100.0)
    y = Omelette.add_predictor(model, f, x)
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
    f = Omelette.ReLUSOS1()
    y = Omelette.add_predictor(model, f, x)
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
    y = Omelette.add_predictor(model, Omelette.ReLUSOS1(), x)
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
    f = Omelette.ReLUQuadratic()
    y = Omelette.add_predictor(model, f, x)
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

end

ReLUTests.runtests()
