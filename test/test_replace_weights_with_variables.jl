# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module TestReplaceWeightsWithVariables

using JuMP
using Test

import MathOptAI

is_test(x) = startswith(string(x), "test_")

function runtests()
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

function test_ReLU()
    model = Model()
    predictor = MathOptAI.ReLU()
    @test predictor ==
          MathOptAI.replace_weights_with_variables(model, predictor)
    return
end

function test_Affine()
    model = Model()
    predictor = MathOptAI.Affine([1.0 3.0; 2.0 4.0], [5.0, 6.0])
    p = MathOptAI.replace_weights_with_variables(model, predictor)
    @test all_variables(model) == vcat(vec(p.A), p.b)
    @test start_value.(all_variables(model)) == 1:6
    return
end

function test_Pipeline()
    model = Model()
    predictor = MathOptAI.Pipeline(
        MathOptAI.Affine([1.0 3.0; 2.0 4.0], [5.0, 6.0]),
        MathOptAI.ReLU(),
        MathOptAI.Scale([7.0, 8.0], [9.0, 10.0]),
    )
    p = MathOptAI.replace_weights_with_variables(model, predictor)
    @test num_variables(model) == 10
    @test start_value.(all_variables(model)) == 1:10
    return
end

function test_Scale()
    model = Model()
    predictor = MathOptAI.Scale([1.0, 2.0], [3.0, 4.0])
    p = MathOptAI.replace_weights_with_variables(model, predictor)
    @test all_variables(model) == vcat(vec(p.scale), p.bias)
    @test start_value.(all_variables(model)) == 1:4
    return
end

end  # module

TestReplaceWeightsWithVariables.runtests()
