# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module LinearRegressionTests

using Test
using JuMP
import Omelette

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$name" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function test_LinearRegression()
    model = Model()
    @variable(model, x[1:2])
    @variable(model, y[1:1])
    f = Omelette.LinearRegression([2.0, 3.0])
    Omelette.add_model(model, f, x, y)
    cons = all_constraints(model; include_variable_in_set_constraints = false)
    obj = constraint_object(only(cons))
    @test obj.set == MOI.EqualTo(0.0)
    @test isequal_canonical(obj.func, 2.0 * x[1] + 3.0 * x[2] - y[1])
    return
end

function test_LinearRegression_dimension_mismatch()
    model = Model()
    @variable(model, x[1:3])
    @variable(model, y[1:2])
    f = Omelette.LinearRegression([2.0, 3.0])
    @test size(f) == (1, 2)
    @test_throws DimensionMismatch Omelette.add_model(model, f, x, y[1:1])
    @test_throws DimensionMismatch Omelette.add_model(model, f, x[1:2], y)
    g = Omelette.LinearRegression([2.0 3.0; 4.0 5.0; 6.0 7.0])
    @test size(g) == (3, 2)
    @test_throws DimensionMismatch Omelette.add_model(model, g, x, y)
    return
end

end

LinearRegressionTests.runtests()
