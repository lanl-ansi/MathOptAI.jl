# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module TestExaModelsExt

using Test

import ExaModels
import MathOptAI
import NLPModelsIpopt

is_test(x) = startswith(string(x), "test_")

function runtests()
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

# ── Helper ────────────────────────────────────────────────────────────────────

function _make_core_with_input(n)
    core = ExaModels.ExaCore()
    x = ExaModels.variable(core, n)
    return core, x
end

# ── Affine ────────────────────────────────────────────────────────────────────

function test_Affine_structure()
    A = [1.0 2.0; 3.0 4.0]
    b = [0.5, -0.5]
    p = MathOptAI.Affine(A, b)
    core, x = _make_core_with_input(2)
    y, form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 4   # 2 inputs + 2 outputs
    @test m.meta.ncon == 2   # one equality per output row
    @test form isa MathOptAI.Formulation
    @test form.predictor === p
    @test length(form.variables) == 1
    @test form.variables[1] === y
    @test length(form.constraints) == 1
    return
end

function test_Affine_end_to_end()
    # min  y[1]^2 + y[2]^2
    # s.t. y = A*x + b,  A = I, b = [1, 2]
    # Optimal: x = [-1, -2], y = [0, 0], obj = 0
    A = [1.0 0.0; 0.0 1.0]
    b = [1.0, 2.0]
    p = MathOptAI.Affine(A, b)
    core = ExaModels.ExaCore()
    x = ExaModels.variable(core, 2)
    y, _ = MathOptAI.add_predictor(core, p, x)
    ExaModels.objective(core, y[i]^2 for i in 1:2)
    m = ExaModels.ExaModel(core)
    result = NLPModelsIpopt.ipopt(m; print_level = 0)
    @test result.status ∈ (:first_order, :acceptable)
    sol = result.solution
    x_opt = sol[1:2]
    y_opt = sol[3:4]
    @test isapprox(y_opt, [0.0, 0.0]; atol = 1.0e-5)
    @test isapprox(x_opt, [-1.0, -2.0]; atol = 1.0e-5)
    return
end

# ── ReducedSpace{Affine} ──────────────────────────────────────────────────────

function test_ReducedSpace_Affine_structure()
    A = [2.0 0.0; 0.0 3.0]
    b = [1.0, -1.0]
    p = MathOptAI.ReducedSpace(MathOptAI.Affine(A, b))
    core, x = _make_core_with_input(2)
    y, form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 2   # only the 2 inputs, no new variables
    @test m.meta.ncon == 0
    @test form isa MathOptAI.Formulation
    @test form.predictor === p
    @test isempty(form.variables)
    @test isempty(form.constraints)
    @test y isa AbstractVector
    @test length(y) == 2
    return
end

# ── Scale ─────────────────────────────────────────────────────────────────────

function test_Scale_structure()
    p = MathOptAI.Scale([2.0, 3.0], [1.0, -1.0])
    core, x = _make_core_with_input(2)
    y, form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 4
    @test m.meta.ncon == 2
    @test form isa MathOptAI.Formulation
    @test form.predictor === p
    return
end

function test_ReducedSpace_Scale_structure()
    p = MathOptAI.ReducedSpace(MathOptAI.Scale([2.0, 3.0], [1.0, -1.0]))
    core, x = _make_core_with_input(2)
    y, form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 2
    @test m.meta.ncon == 0
    @test isempty(form.variables)
    @test isempty(form.constraints)
    @test length(y) == 2
    return
end

# ── ReLU ──────────────────────────────────────────────────────────────────────

function test_ReLU_structure()
    p = MathOptAI.ReLU()
    core, x = _make_core_with_input(3)
    y, form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 6   # 3 inputs + 3 outputs
    @test m.meta.ncon == 3
    @test all(m.meta.lvar[4:6] .== 0.0)   # output bounded below by 0
    @test form isa MathOptAI.Formulation
    return
end

function test_ReducedSpace_ReLU_structure()
    p = MathOptAI.ReducedSpace(MathOptAI.ReLU())
    core, x = _make_core_with_input(3)
    y, form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 3
    @test m.meta.ncon == 0
    @test length(y) == 3
    return
end

function test_ReLU_AbstractVector()
    p = MathOptAI.ReLU()
    core, x = _make_core_with_input(3)
    y, form = MathOptAI.add_predictor(core, p, [x[i] for i in 1:3])
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 6   # 3 inputs + 3 outputs
    @test m.meta.ncon == 3
    @test all(m.meta.lvar[4:6] .== 0.0)   # output bounded below by 0
    @test form isa MathOptAI.Formulation
    return
end

# ── Sigmoid ───────────────────────────────────────────────────────────────────

function test_Sigmoid_structure()
    p = MathOptAI.Sigmoid()
    core, x = _make_core_with_input(2)
    y, form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 4
    @test m.meta.ncon == 2
    @test all(m.meta.lvar[3:4] .== 0.0)
    @test all(m.meta.uvar[3:4] .== 1.0)
    @test form isa MathOptAI.Formulation
    return
end

function test_ReducedSpace_Sigmoid_structure()
    p = MathOptAI.ReducedSpace(MathOptAI.Sigmoid())
    core, x = _make_core_with_input(2)
    y, form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 2
    @test m.meta.ncon == 0
    @test length(y) == 2
    return
end

function test_Sigmoid_structure()
    p = MathOptAI.Sigmoid()
    core, x = _make_core_with_input(2)
    y, form = MathOptAI.add_predictor(core, p, [x[i] for i in 1:2])
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 4
    @test m.meta.ncon == 2
    @test all(m.meta.lvar[3:4] .== 0.0)
    @test all(m.meta.uvar[3:4] .== 1.0)
    @test form isa MathOptAI.Formulation
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

# ── Tanh ──────────────────────────────────────────────────────────────────────

function test_Tanh_structure()
    p = MathOptAI.Tanh()
    core, x = _make_core_with_input(2)
    y, form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 4
    @test m.meta.ncon == 2
    @test all(m.meta.lvar[3:4] .== -1.0)
    @test all(m.meta.uvar[3:4] .== 1.0)
    @test form isa MathOptAI.Formulation
    return
end

function test_ReducedSpace_Tanh_structure()
    p = MathOptAI.ReducedSpace(MathOptAI.Tanh())
    core, x = _make_core_with_input(2)
    y, form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 2
    @test m.meta.ncon == 0
    @test length(y) == 2
    return
end

function test_Tanh_AbstractVector()
    p = MathOptAI.Tanh()
    core, x = _make_core_with_input(2)
    y, form = MathOptAI.add_predictor(core, p, [x[i] for i in 1:2])
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 4
    @test m.meta.ncon == 2
    @test all(m.meta.lvar[3:4] .== -1.0)
    @test all(m.meta.uvar[3:4] .== 1.0)
    @test form isa MathOptAI.Formulation
    return
end

# ── SoftPlus ──────────────────────────────────────────────────────────────────

function test_SoftPlus_structure()
    p = MathOptAI.SoftPlus()
    core, x = _make_core_with_input(2)
    y, form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 4
    @test m.meta.ncon == 2
    @test all(m.meta.lvar[3:4] .== 0.0)
    @test form isa MathOptAI.Formulation
    return
end

function test_ReducedSpace_SoftPlus_structure()
    p = MathOptAI.ReducedSpace(MathOptAI.SoftPlus())
    core, x = _make_core_with_input(2)
    y, form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 2
    @test m.meta.ncon == 0
    @test length(y) == 2
    return
end

function test_SoftPlus_AbstractVector()
    p = MathOptAI.SoftPlus()
    core, x = _make_core_with_input(2)
    y, form = MathOptAI.add_predictor(core, p, [x[i] for i in 1:2])
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 4
    @test m.meta.ncon == 2
    @test all(m.meta.lvar[3:4] .== 0.0)
    @test form isa MathOptAI.Formulation
    return
end

# ── GELU ──────────────────────────────────────────────────────────────────────

function test_GELU_structure()
    p = MathOptAI.GELU()
    core, x = _make_core_with_input(2)
    y, form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 4
    @test m.meta.ncon == 2
    @test form isa MathOptAI.Formulation
    return
end

function test_ReducedSpace_GELU_structure()
    p = MathOptAI.ReducedSpace(MathOptAI.GELU())
    core, x = _make_core_with_input(2)
    y, form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 2
    @test m.meta.ncon == 0
    @test length(y) == 2
    return
end

function test_GELU_AbstractVector()
    p = MathOptAI.GELU()
    core, x = _make_core_with_input(2)
    y, form = MathOptAI.add_predictor(core, p, [x[i] for i in 1:2])
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 4
    @test m.meta.ncon == 2
    @test form isa MathOptAI.Formulation
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

# ── LeakyReLU ────────────────────────────────────────────────────────────────

function test_LeakyReLU_structure()
    p = MathOptAI.LeakyReLU(; negative_slope = 0.01)
    core, x = _make_core_with_input(3)
    y, form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 9   # 3 input + 3 relu + 3 leaky
    @test m.meta.ncon == 6   # 3 relu + 3 leaky
    @test form isa MathOptAI.Formulation
    @test form.predictor === p
    return
end

function test_ReducedSpace_LeakyReLU_structure()
    p = MathOptAI.ReducedSpace(MathOptAI.LeakyReLU(; negative_slope = 0.01))
    core, x = _make_core_with_input(3)
    y, form = MathOptAI.add_predictor(core, p, x)
    @test length(y) == 3
    @test form.predictor isa MathOptAI.ReducedSpace{<:MathOptAI.LeakyReLU}
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 3
    @test m.meta.ncon == 0
    return
end

function test_LeakyReLU_AbstractVector()
    p = MathOptAI.LeakyReLU(; negative_slope = 0.01)
    core, x = _make_core_with_input(3)
    y, form = MathOptAI.add_predictor(core, p, [x[i] for i in 1:3])
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 9   # 3 input + 3 relu + 3 leaky
    @test m.meta.ncon == 6   # 3 relu + 3 leaky
    @test form isa MathOptAI.Formulation
    @test form.predictor === p
    return
end

# ── Permutation ──────────────────────────────────────────────────────────────

function test_Permutation_structure()
    perm = MathOptAI.Permutation([3, 1, 2])
    core, x = _make_core_with_input(3)
    y, form = MathOptAI.add_predictor(core, MathOptAI.ReducedSpace(perm), x)
    @test length(y) == 3
    @test form.predictor isa MathOptAI.ReducedSpace{MathOptAI.Permutation}
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 3
    @test m.meta.ncon == 0
    return
end

# ── SoftMax ──────────────────────────────────────────────────────────────────

function test_SoftMax_structure()
    p = MathOptAI.SoftMax()
    core, x = _make_core_with_input(3)
    y, form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 7   # 3 input + 1 denom + 3 y
    @test m.meta.ncon == 4   # 1 denom + 3 y
    @test form isa MathOptAI.Formulation
    @test form.predictor === p
    return
end

function test_ReducedSpace_SoftMax_structure()
    p = MathOptAI.ReducedSpace(MathOptAI.SoftMax())
    core, x = _make_core_with_input(3)
    y, form = MathOptAI.add_predictor(core, p, x)
    @test length(y) == 3
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 4   # 3 input + 1 denom
    @test m.meta.ncon == 1   # 1 denom
    return
end

function test_SoftMax_AbstractVector()
    p = MathOptAI.SoftMax()
    core, x = _make_core_with_input(3)
    y, form = MathOptAI.add_predictor(core, p, [x[i] for i in 1:3])
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 7   # 3 input + 1 denom + 3 y
    @test m.meta.ncon == 4   # 1 denom + 3 y
    @test form isa MathOptAI.Formulation
    @test form.predictor === p
    return
end

# ── Pipeline ──────────────────────────────────────────────────────────────────

function test_Pipeline_structure()
    p = MathOptAI.Pipeline(
        MathOptAI.Affine([1.0 0.0; 0.0 1.0], [0.0, 0.0]),
        MathOptAI.ReLU(),
        MathOptAI.Affine([1.0 0.0; 0.0 1.0], [0.0, 0.0]),
    )
    core, x = _make_core_with_input(2)
    y, form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    # x(2) + Affine_out(2) + ReLU_out(2) + Affine_out(2) = 8 vars
    @test m.meta.nvar == 8
    # Affine(2 cons) + ReLU(2 cons) + Affine(2 cons) = 6 cons
    @test m.meta.ncon == 6
    @test form isa MathOptAI.PipelineFormulation
    @test length(form.layers) == 3
    return
end

function test_ReducedSpace_Pipeline_structure()
    p = MathOptAI.ReducedSpace(
        MathOptAI.Pipeline(
            MathOptAI.Affine([2.0 0.0; 0.0 3.0], [1.0, -1.0]),
            MathOptAI.ReLU(),
        ),
    )
    core, x = _make_core_with_input(2)
    y, form = MathOptAI.add_predictor(core, p, x)
    m = ExaModels.ExaModel(core)
    @test m.meta.nvar == 2
    @test m.meta.ncon == 0
    @test length(y) == 2
    @test form isa MathOptAI.PipelineFormulation
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
    core = ExaModels.ExaCore()
    x = ExaModels.variable(core, 1; start = 1.0)
    y, _ = MathOptAI.add_predictor(core, p, x)
    ExaModels.objective(core, y[i]^2 for i in 1:1)
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

# ── GrayBox (error) ───────────────────────────────────────────────────────────

function test_GrayBox_error()
    core = ExaModels.ExaCore()
    x = ExaModels.variable(core, 2)
    @test_throws(
        ErrorException(
            "GrayBox is not supported with ExaCore. Convert your model to a " *
            "Pipeline of explicit layer predictors.",
        ),
        MathOptAI.add_predictor(core, MathOptAI.GrayBox(identity), x),
    )
    return
end

end  # module

TestExaModelsExt.runtests()
