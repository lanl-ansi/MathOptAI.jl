# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module TestInfiniteOpt

using JuMP
using Test

import Distributions
import InfiniteOpt
import Ipopt
import MathOptAI

is_test(x) = startswith(string(x), "test_")

function runtests()
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

function MathOptAI.add_variables(
    model::JuMP.AbstractModel,
    predictor::MathOptAI.AbstractPredictor,
    x::Vector{<:InfiniteOpt.GeneralVariableRef},
    n::Int,
    base_name::String,
)
    params = InfiniteOpt.parameter_refs.(x)
    if all(isempty, params)
        return @variable(model, [1:n], base_name = base_name)
    end
    @assert length(unique(params)) == 1
    return @variable(
        model,
        [1:n],
        base_name = base_name,
        variable_type = InfiniteOpt.Infinite(first(params)...),
    )
end

function test_extension()
    predictor = MathOptAI.Tanh()
    model = InfiniteOpt.InfiniteModel(Ipopt.Optimizer)
    set_silent(model)
    InfiniteOpt.@infinite_parameter(model, ξ ~ Distributions.Uniform(0, 1))
    @variable(model, -1 <= x <= 3, InfiniteOpt.Infinite(ξ))
    y, _ = MathOptAI.add_predictor(model, predictor, [x])
    @objective(model, Max, InfiniteOpt.𝔼(only(y), ξ))
    @constraint(model, x <= ξ)
    optimize!(model)
    @test MathOptAI.get_variable_bounds(y[1]) == (tanh(-1), tanh(3))
    y_v = value(only(y))
    @test isapprox(y_v, tanh.(value(x)); atol = 1e-5)
    @test isapprox(objective_value(model), sum(y_v) / length(y_v); atol = 1e-5)
    return
end

end  # module

TestInfiniteOpt.runtests()
