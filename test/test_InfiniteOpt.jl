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
        return JuMP.@variable(model, [1:n], base_name = base_name)
    end
    @assert length(unique(params)) == 1
    return JuMP.@variable(
        model,
        [1:n],
        base_name = base_name,
        variable_type = InfiniteOpt.Infinite(first(params)...),
    )
end

function test_extension()
    predictor = MathOptAI.Pipeline(
        MathOptAI.Affine(Float64[1 2 3; 4 5 6]),
        MathOptAI.ReLU(),
    )
    model = InfiniteOpt.InfiniteModel()
    InfiniteOpt.@infinite_parameter(model, ξ ~ Distributions.Uniform(0, 1))
    JuMP.@variable(model, 1 <= x[1:2] <= 3, InfiniteOpt.Infinite(ξ))
    y = MathOptAI.add_predictor(model, predictor, x)
    return
end

end  # module

TestInfiniteOpt.runtests()
