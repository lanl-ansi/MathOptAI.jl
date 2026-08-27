# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module TestExaModelsExt

using Test

import ExaModels
import Flux
import MathOptAI
import NLPModelsIpopt

is_test(x) = startswith(string(x), "test_")

function runtests()
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

function _test_solution(fn, core, x, y::ExaModels.Variable)
    model = ExaModels.ExaModel(core)
    result = NLPModelsIpopt.ipopt(model; print_level = 0)
    @test result.status ∈ (:first_order, :acceptable)
    @test isapprox(
        ExaModels.solution(result, y),
        fn(ExaModels.solution(result, x));
        atol = 1e-6,
    )
    return
end

function _test_solution(fn, core, x, y::ExaModels.Expression)
    core, z = ExaModels.add_var(core, y.length)
    core, _ = ExaModels.add_con(
        core,
        y[i] - z[i] for i in 1:y.length;
        lcon = 0,
        ucon = 0,
    )
    _test_solution(fn, core, x, z)
    return
end

function test_Affine_structure()
    A = [1.0 2.0; 3.0 4.0]
    b = [0.5, -0.5]
    p = MathOptAI.Affine(A, b)
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-1.0, 2.0]
    core, x = ExaModels.add_var(core, 2; lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 4   # 2 inputs + 2 outputs
    @test m.meta.ncon == 2   # one equality per output row
    @test form isa MathOptAI.Formulation
    @test form.predictor === p
    @test length(form.variables) == 1
    @test form.variables[1] === y
    @test length(form.constraints) == 1
    _test_solution(x -> A * x + b, core, x, y)
    return
end

function test_Affine_reduced_space_false()
    A = [1.0 2.0; 3.0 4.0]
    b = [0.5, -0.5]
    p = MathOptAI.Affine(A, b)
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-1.0, 2.0]
    core, x = ExaModels.add_var(core, 2; lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x; reduced_space = false)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 4   # 2 inputs + 2 outputs
    @test m.meta.ncon == 2   # one equality per output row
    @test form isa MathOptAI.Formulation
    @test form.predictor === p
    @test length(form.variables) == 1
    @test form.variables[1] === y
    @test length(form.constraints) == 1
    _test_solution(x -> A * x + b, core, x, y)
    return
end

function test_Affine_end_to_end()
    # min  y[1]^2 + y[2]^2
    # s.t. y = A*x + b,  A = I, b = [1, 2]
    # Optimal: x = [-1, -2], y = [0, 0], obj = 0
    A = [1.0 0.0; 0.0 1.0]
    b = [1.0, 2.0]
    p = MathOptAI.Affine(A, b)
    core = ExaModels.ExaCore(; concrete = Val(true))
    core, x = ExaModels.add_var(core, 2)
    (core, y), _ = MathOptAI.add_predictor(core, p, x)
    core, _ = ExaModels.add_obj(core, y[i]^2 for i in 1:2)
    m = ExaModels.ExaModel(core)
    result = NLPModelsIpopt.ipopt(m; print_level = 0)
    @test result.status ∈ (:first_order, :acceptable)
    sol = result.solution
    x_opt = sol[1:2]
    y_opt = sol[3:4]
    @test isapprox(y_opt, [0.0, 0.0]; atol = 1.0e-5)
    @test isapprox(x_opt, [-1.0, -2.0]; atol = 1.0e-5)
    _test_solution(x -> A * x + b, core, x, y)
    return
end

function test_ReducedSpace_Affine_structure()
    A = [2.0 0.0; 0.0 3.0]
    b = [1.0, -1.0]
    p = MathOptAI.ReducedSpace(MathOptAI.Affine(A, b))
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-1.0, 2.0]
    core, x = ExaModels.add_var(core, 2; lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 2   # only the 2 inputs, no new variables
    @test m.meta.ncon == 0
    @test form isa MathOptAI.Formulation
    @test form.predictor === p
    @test isempty(form.variables)
    @test isempty(form.constraints)
    @test y isa ExaModels.Expression
    @test y.length == 2
    _test_solution(x -> A * x + b, core, x, y)
    return
end

function test_ReducedSpace_Affine_kwarg()
    A = [2.0 0.0; 0.0 3.0]
    b = [1.0, -1.0]
    p = MathOptAI.ReducedSpace(MathOptAI.Affine(A, b))
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-1.0, 2.0]
    core, x = ExaModels.add_var(core, 2; lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x; reduced_space = true)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 2   # only the 2 inputs, no new variables
    @test m.meta.ncon == 0
    @test form isa MathOptAI.Formulation
    @test form.predictor === p
    @test isempty(form.variables)
    @test isempty(form.constraints)
    @test y isa ExaModels.Expression
    @test y.length == 2
    _test_solution(x -> A * x + b, core, x, y)
    return
end

function test_Scale_structure()
    p = MathOptAI.Scale([2.0, 3.0], [1.0, -1.0])
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-1.0, 2.0]
    core, x = ExaModels.add_var(core, 2; lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 4
    @test m.meta.ncon == 2
    @test form isa MathOptAI.Formulation
    @test form.predictor === p
    _test_solution(x -> p.scale .* x .+ p.bias, core, x, y)
    return
end

function test_ReducedSpace_Scale_structure()
    p = MathOptAI.ReducedSpace(MathOptAI.Scale([2.0, 3.0], [1.0, -1.0]))
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-1.0, 2.0]
    core, x = ExaModels.add_var(core, 2; lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 2
    @test m.meta.ncon == 0
    @test isempty(form.variables)
    @test isempty(form.constraints)
    @test y isa ExaModels.Expression
    @test y.length == 2
    _test_solution(x -> p.predictor.scale .* x .+ p.predictor.bias, core, x, y)
    return
end

function test_ReLU_structure()
    p = MathOptAI.ReLU()
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-1.0, 0.5, 2.0]
    core, x = ExaModels.add_var(core, 3; lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 6   # 3 inputs + 3 outputs
    @test m.meta.ncon == 3
    @test all(m.meta.lvar[4:6] .== 0.0)   # output bounded below by 0
    @test form isa MathOptAI.Formulation
    _test_solution(x -> max.(0, x), core, x, y)
    return
end

function test_ReducedSpace_ReLU_structure()
    p = MathOptAI.ReducedSpace(MathOptAI.ReLU())
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-1.0, 0.5, 2.0]
    core, x = ExaModels.add_var(core, 3; lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 3
    @test m.meta.ncon == 0
    @test y isa ExaModels.Expression
    @test y.length == 3
    _test_solution(x -> max.(0, x), core, x, y)
    return
end

function test_ReLUEpigraph_structure()
    p = MathOptAI.ReLUEpigraph()
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-1.0, 0.5, 2.0]
    core, x = ExaModels.add_var(core, 3; lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 6   # 3 inputs + 3 outputs
    @test m.meta.ncon == 3
    @test all(m.meta.lvar[4:6] .== 0.0)   # output bounded below by 0
    @test form isa MathOptAI.Formulation
    core, _ = ExaModels.add_obj(core, y[i] for i in 1:3)
    _test_solution(x -> max.(0, x), core, x, y)
    return
end

function test_ReLUEpigraph_end_to_end()
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-2.0, 1.5, 3.0]
    core, x = ExaModels.add_var(core, 3; lvar = x0, uvar = x0)
    p = MathOptAI.ReLUEpigraph()
    (core, y), _ = MathOptAI.add_predictor(core, p, x)
    core, _ = ExaModels.add_obj(core, y[i] for i in 1:3)
    _test_solution(x -> max.(0, x), core, x, y)
    return
end

function test_Sigmoid_structure()
    p = MathOptAI.Sigmoid()
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-2.0, 1.5]
    core, x = ExaModels.add_var(core, 2; lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 4
    @test m.meta.ncon == 2
    @test all(m.meta.lvar[3:4] .== 0.0)
    @test all(m.meta.uvar[3:4] .== 1.0)
    @test form isa MathOptAI.Formulation
    _test_solution(x -> p.(x), core, x, y)
    return
end

function test_ReducedSpace_Sigmoid_structure()
    p = MathOptAI.ReducedSpace(MathOptAI.Sigmoid())
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-2.0, 1.5]
    core, x = ExaModels.add_var(core, 2; lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 2
    @test m.meta.ncon == 0
    @test y isa ExaModels.Expression
    @test y.length == 2
    _test_solution(x -> p.predictor.(x), core, x, y)
    return
end

function test_Sigmoid_derivative_correctness()
    ext = Base.get_extension(MathOptAI, :MathOptAIExaModelsExt)
    for xv in [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]
        h = 1e-6
        @test isapprox(
            ext._d_sigmoid(xv),
            (ext._sigmoid(xv + h) - ext._sigmoid(xv - h)) / (2h);
            atol = 1.0e-6,
            rtol = 1.0e-4,
        )
        @test isapprox(
            ext._dd_sigmoid(xv),
            (ext._d_sigmoid(xv + h) - ext._d_sigmoid(xv - h)) / (2h);
            atol = 1.0e-6,
            rtol = 1.0e-4,
        )
    end
    return
end

function test_Tanh_structure()
    p = MathOptAI.Tanh()
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-2.0, 1.5]
    core, x = ExaModels.add_var(core, 2; lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 4
    @test m.meta.ncon == 2
    @test all(m.meta.lvar[3:4] .== -1.0)
    @test all(m.meta.uvar[3:4] .== 1.0)
    @test form isa MathOptAI.Formulation
    _test_solution(x -> p.(x), core, x, y)
    return
end

function test_ReducedSpace_Tanh_structure()
    p = MathOptAI.ReducedSpace(MathOptAI.Tanh())
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-2.0, 1.5]
    core, x = ExaModels.add_var(core, 2; lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 2
    @test m.meta.ncon == 0
    @test y isa ExaModels.Expression
    @test y.length == 2
    _test_solution(x -> p.predictor.(x), core, x, y)
    return
end

function test_SoftPlus_structure()
    p = MathOptAI.SoftPlus()
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-2.0, 1.5]
    core, x = ExaModels.add_var(core, 2; lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 4
    @test m.meta.ncon == 2
    @test all(m.meta.lvar[3:4] .== 0.0)
    @test form isa MathOptAI.Formulation
    _test_solution(x -> p.(x), core, x, y)
    return
end

function test_ReducedSpace_SoftPlus_structure()
    p = MathOptAI.ReducedSpace(MathOptAI.SoftPlus())
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-2.0, 1.5]
    core, x = ExaModels.add_var(core, 2; lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 2
    @test m.meta.ncon == 0
    @test y isa ExaModels.Expression
    @test y.length == 2
    _test_solution(x -> p.predictor.(x), core, x, y)
    return
end

function test_GELU_structure()
    p = MathOptAI.GELU()
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-2.0, 1.5]
    core, x = ExaModels.add_var(core, 2; lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 4
    @test m.meta.ncon == 2
    @test form isa MathOptAI.Formulation
    _test_solution(x -> p.(x), core, x, y)
    return
end

function test_ReducedSpace_GELU_structure()
    p = MathOptAI.ReducedSpace(MathOptAI.GELU())
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-2.0, 1.5]
    core, x = ExaModels.add_var(core, 2; lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 2
    @test m.meta.ncon == 0
    @test y isa ExaModels.Expression
    @test y.length == 2
    _test_solution(x -> p.predictor.(x), core, x, y)
    return
end

function test_GELU_derivative_correctness()
    # Verify the registered GELU second derivative matches finite difference
    ext = Base.get_extension(MathOptAI, :MathOptAIExaModelsExt)
    for xv in [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]
        h = 1e-6
        @test isapprox(
            ext._d_gelu(xv),
            (ext._gelu(xv + h) - ext._gelu(xv - h)) / (2h);
            atol = 1.0e-6,
            rtol = 1.0e-4,
        )
        @test isapprox(
            ext._dd_gelu(xv),
            (ext._d_gelu(xv + h) - ext._d_gelu(xv - h)) / (2h);
            atol = 1.0e-6,
            rtol = 1.0e-4,
        )
    end
    return
end

function test_GELU_end_to_end()
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-2.0, 1.5, 3.0]
    core, x = ExaModels.add_var(core, 3; lvar = x0, uvar = x0)
    p = MathOptAI.GELU()
    (core, y), _ = MathOptAI.add_predictor(core, p, x)
    core, _ = ExaModels.add_obj(core, y[i] for i in 1:3)
    _test_solution(x -> p.(x), core, x, y)
    return
end

function test_LeakyReLU_structure()
    p = MathOptAI.LeakyReLU(; negative_slope = 0.01)
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-2.0, 1.5, 3.0]
    core, x = ExaModels.add_var(core, 3; lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 9   # 3 input + 3 relu + 3 leaky
    @test m.meta.ncon == 6   # 3 relu + 3 leaky
    @test form isa MathOptAI.Formulation
    @test form.predictor === p
    _test_solution(x -> p.(x), core, x, y)
    return
end

function test_ReducedSpace_LeakyReLU_structure()
    p = MathOptAI.ReducedSpace(MathOptAI.LeakyReLU(; negative_slope = 0.01))
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-2.0, 1.5, 3.0]
    core, x = ExaModels.add_var(core, 3; lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x)
    @test y isa ExaModels.Expression
    @test y.length == 3
    @test form.predictor isa MathOptAI.ReducedSpace{<:MathOptAI.LeakyReLU}
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 3
    @test m.meta.ncon == 0
    _test_solution(x -> p.predictor.(x), core, x, y)
    return
end

function test_LeakyReLU_end_to_end()
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-2.0, 1.5, 3.0]
    core, x = ExaModels.add_var(core, 3; lvar = x0, uvar = x0)
    p = MathOptAI.LeakyReLU(; negative_slope = 0.01)
    (core, y), _ = MathOptAI.add_predictor(core, p, x)
    _test_solution(x -> p.(x), core, x, y)
    return
end

function test_Permutation_structure()
    perm = MathOptAI.Permutation([3, 1, 2])
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-2.0, 1.5, 3.0]
    core, x = ExaModels.add_var(core, 3; lvar = x0, uvar = x0)
    (core, y), form =
        MathOptAI.add_predictor(core, MathOptAI.ReducedSpace(perm), x)
    @test y isa ExaModels.Expression
    @test y.length == 3
    @test y[1] == x[3]
    @test y[2] == x[1]
    @test y[3] == x[2]
    @test form.predictor isa MathOptAI.ReducedSpace{MathOptAI.Permutation}
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 3
    @test m.meta.ncon == 0
    # https://github.com/madsuite-org/ExaModels.jl/issues/293
    # _test_solution(x -> x[perm.p], core, x, y)
    return
end

function test_SoftMax_structure()
    p = MathOptAI.SoftMax()
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-1.0, 2.0, 3.0]
    core, x = ExaModels.add_var(core, 3; lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 7   # 3 input + 1 denom + 3 y
    @test m.meta.ncon == 4   # 1 denom + 3 y
    @test form isa MathOptAI.Formulation
    @test form.predictor === p
    _test_solution(x -> p(collect(x)), core, x, y)
    return
end

function test_Pipeline_structure()
    p = MathOptAI.Pipeline(
        MathOptAI.Affine([1.0 0.0; 0.0 1.0], [0.0, 0.0]),
        MathOptAI.ReLU(),
        MathOptAI.Affine([1.0 0.0; 0.0 1.0], [0.0, 0.0]),
    )
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-2.0, 1.5]
    core, x = ExaModels.add_var(core, 2; lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    # x(2) + Affine_out(2) + ReLU_out(2) + Affine_out(2) = 8 vars
    @test m.meta.nvar == 8
    # Affine(2 cons) + ReLU(2 cons) + Affine(2 cons) = 6 cons
    @test m.meta.ncon == 6
    @test form isa MathOptAI.PipelineFormulation
    @test length(form.layers) == 3
    _test_solution(x -> max.(0, p.layers[1].A * x + p.layers[1].b), core, x, y)
    return
end

function test_ReducedSpace_Pipeline_structure()
    aff = MathOptAI.Affine([2.0 0.0; 0.0 3.0], [1.0, -1.0])
    p = MathOptAI.ReducedSpace(MathOptAI.Pipeline(aff, MathOptAI.ReLU()))
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-2.0, 1.5]
    core, x = ExaModels.add_var(core, 2; lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 2
    @test m.meta.ncon == 0
    @test y isa ExaModels.Expression
    @test y.length == 2
    @test form isa MathOptAI.PipelineFormulation
    _test_solution(x -> max.(0, aff.A * x + aff.b), core, x, y)
    return
end

function test_Pipeline_end_to_end()
    # Pipeline: Affine(1→1) → ReLU → Affine(1→1)
    # Layer 1: y1 = x        (A=[[1]], b=[0])
    # ReLU:    y2 = max(0, x)
    # Layer 2: y3 = y2 - 0.5 (A=[[1]], b=[-0.5])
    # Objective: y3^2 = (max(0,x) - 0.5)^2
    # Optimal: x = 0.5, y1 = 0.5, y2 = 0.5, y3 = 0, obj = 0
    p = MathOptAI.Pipeline(
        MathOptAI.Affine(reshape([1.0], 1, 1), [0.0]),
        MathOptAI.ReLU(),
        MathOptAI.Affine(reshape([1.0], 1, 1), [-0.5]),
    )
    core = ExaModels.ExaCore(; concrete = Val(true))
    core, x = ExaModels.add_var(core, 1; start = 1.0)
    (core, y), _ = MathOptAI.add_predictor(core, p, x)
    core, _ = ExaModels.add_obj(core, y[i]^2 for i in 1:1)
    m = ExaModels.ExaModel(core)
    result = NLPModelsIpopt.ipopt(m; print_level = 0)
    @test result.status ∈ (:first_order, :acceptable)
    sol = result.solution
    # y3 (last variable, index 4) should be 0 at optimal
    y3_val = sol[end]
    @test isapprox(y3_val, 0.0; atol = 1.0e-5)
    # x (index 1) should be 0.5 at optimal
    x_val = sol[1]
    @test isapprox(x_val, 0.5; atol = 1.0e-4)
    return
end

function test_flux_end_to_end()
    chain = Flux.Chain(
        Flux.Dense(2 => 2, Flux.relu),
        Flux.Scale(2),
        Flux.Dense(2 => 2, Flux.sigmoid),
        Flux.softmax,
        Flux.Dense(2 => 2, Flux.softplus),
        Flux.Dense(2 => 2, Flux.tanh),
    )
    core = ExaModels.ExaCore(; concrete = Val(true))
    b = [1.1, 2.3]
    core, x = ExaModels.add_var(core, 2; lvar = b, uvar = b)
    (core, y), _ = MathOptAI.add_predictor(core, chain, x)
    _test_solution(x -> chain(Float32.(x)), core, x, y)
    return
end

function test_flux_end_to_end_gray_box()
    chain = Flux.Chain(
        Flux.Dense(2 => 2, Flux.relu),
        Flux.Scale(2),
        Flux.Dense(2 => 2, Flux.sigmoid),
        Flux.softmax,
        Flux.Dense(2 => 2, Flux.softplus),
        Flux.Dense(2 => 3, Flux.tanh),
    )
    core = ExaModels.ExaCore(; concrete = Val(true))
    b = [1.1, 2.3]
    core, x = ExaModels.add_var(core, 2; lvar = b, uvar = b)
    (core, y), _ = MathOptAI.add_predictor(core, chain, x; gray_box = true)
    _test_solution(x -> chain(Float32.(x)), core, x, y)
    return
end

function test_SoftPlusEpigraph_AbstractVariable()
    p = MathOptAI.SoftPlusEpigraph(; beta = 2.0)
    core = ExaModels.ExaCore(; concrete = Val(true))
    x0 = [-1.0, 2.0]
    core, x = ExaModels.add_var(core, length(x0); lvar = x0, uvar = x0)
    (core, y), form = MathOptAI.add_predictor(core, p, x)
    core, _ = ExaModels.add_obj(core, y[i] for i in 1:2)
    _test_solution(x -> MathOptAI.SoftPlus(; beta = 2.0).(x), core, x, y)
    return
end

function test_ReducedSpace_SoftMax_structure()
    predictor = MathOptAI.ReducedSpace(MathOptAI.SoftMax())
    core = ExaModels.ExaCore(; concrete = Val(true))
    core, x = ExaModels.add_var(core, 3)
    @test_throws(
        ErrorException(
            """
            Unsupported predictor: `$predictor`.

            ExaModels does not support the reduced-space formulation of this \
            predictor.
            """,
        ),
        MathOptAI.add_predictor(core, predictor, x),
    )
    return
end

end  # module

TestExaModelsExt.runtests()
