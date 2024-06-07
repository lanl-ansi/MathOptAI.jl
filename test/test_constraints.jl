# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module ConstraintTests

using JuMP
using Test

import Ipopt
import MathOptAI

is_test(x) = startswith(string(x), "test_")

function runtests()
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

function test_normal_lower_limit()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, 0 <= x <= 5)
    @objective(model, Min, x)
    f = MathOptAI.UnivariateNormalDistribution(;
        mean = x -> only(x),
        std_dev = x -> 1.0,
    )
    MathOptAI.add_constraint(model, f, [x], MOI.Interval(0.5, Inf), 0.95)
    optimize!(model)
    @test is_solved_and_feasible(model)
    # μ: Distributions.invlogcdf(Distributions.Normal(μ, 1.0), log(0.05)) = 0.5
    @test isapprox(value(x), 2.1448536; atol = 1e-4)
    return
end

function test_normal_upper_limit()
    model = Model(Ipopt.Optimizer)
    @variable(model, -5 <= x <= 5)
    @objective(model, Max, x)
    f = MathOptAI.UnivariateNormalDistribution(;
        mean = x -> only(x),
        std_dev = x -> 1.0,
    )
    MathOptAI.add_constraint(model, f, [x], MOI.Interval(-Inf, 0.5), 0.95)
    set_silent(model)
    optimize!(model)
    @test is_solved_and_feasible(model)
    # μ: Distributions.invlogcdf(Distributions.Normal(μ, 1.0), log(0.95)) = 0.5
    @test isapprox(value(x), -1.1448536; atol = 1e-4)
    return
end

end  # module

ConstraintTests.runtests()
