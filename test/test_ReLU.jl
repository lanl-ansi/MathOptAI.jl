# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module ReLUTests

using JuMP
using Test

import GLM
import HiGHS
import Omelette

is_test(x) = startswith(string(x), "test_")

function runtests()
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

function test_ReLU_BigM()
    model = Model()
    @variable(model, x[1:2])
    f = Omelette.ReLUBigM(2, 100.0)
    @test size(f) == (2, 2)
    y = Omelette.add_predictor(model, f, x)
    @test length(y) == 2
    @test num_variables(model) == 6
    @test num_constraints(model, AffExpr, MOI.LessThan{Float64}) == 4
    @test num_constraints(model, AffExpr, MOI.GreaterThan{Float64}) == 4
    return
end

function test_ReLU_SOS1()
    model = Model()
    @variable(model, x[1:2])
    f = Omelette.ReLUSOS1(2)
    @test size(f) == (2, 2)
    y = Omelette.add_predictor(model, f, x)
    @test length(y) == 2
    @test num_variables(model) == 8
    @test num_constraints(model, Vector{VariableRef}, MOI.SOS1{Float64}) == 2
    return
end

function test_ReLU_Quadratic()
    model = Model()
    @variable(model, x[1:2])
    f = Omelette.ReLUQuadratic(2)
    @test size(f) == (2, 2)
    y = Omelette.add_predictor(model, f, x)
    @test length(y) == 2
    @test num_variables(model) == 8
    @test num_constraints(model, QuadExpr, MOI.EqualTo{Float64}) == 2
    return
end

end

ReLUTests.runtests()
