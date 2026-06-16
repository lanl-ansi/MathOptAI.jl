# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.SoftMax,
    x::ExaModels.AbstractVariable,
)
    n = _length(x)
    core, denom = ExaModels.add_var(core, 1; lvar = 0.0)
    core, y = ExaModels.add_var(core, n; lvar = 0.0, uvar = 1.0)
    # denom[1] - sum_j exp(x[j]) = 0
    core, c_denom =
        ExaModels.add_con(core, denom[1] for i in 1:1; lcon = 0.0, ucon = 0.0)
    for j in 1:n
        xj = x[j]
        core, _ = ExaModels.add_con!(core, c_denom, i => -exp(xj) for i in 1:1)
    end
    # y[i] - exp(x[i]) / denom[1] = 0
    d = denom[1]
    core, c_y = ExaModels.add_con(
        core,
        y[i] - exp(x[i]) / d for i in 1:n;
        lcon = 0.0,
        ucon = 0.0,
    )
    return (core, y), MathOptAI.Formulation(p, [denom, y], Any[c_denom, c_y])
end

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.SoftMax,
    x::AbstractVector,
)
    n = length(x)
    core, denom = ExaModels.add_var(core, 1; lvar = 0.0)
    core, y = ExaModels.add_var(core, n; lvar = 0.0, uvar = 1.0)
    denom_expr = denom[1]
    for j in 1:n
        denom_expr = denom_expr - exp(x[j])
    end
    core, c_denom = ExaModels.add_con(core, denom_expr; lcon = 0.0, ucon = 0.0)
    d = denom[1]
    cons_y = Any[]
    for i in 1:n
        core, c = ExaModels.add_con(
            core,
            y[i] - exp(x[i]) / d;
            lcon = 0.0,
            ucon = 0.0,
        )
        push!(cons_y, c)
    end
    return (core, y),
    MathOptAI.Formulation(p, [denom, y], Any[c_denom; cons_y...])
end

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{MathOptAI.SoftMax},
    x,
)
    n = _length(x)
    core, denom = ExaModels.add_var(core, 1; lvar = 0.0)
    core, c_denom =
        ExaModels.add_con(core, denom[1] for i in 1:1; lcon = 0.0, ucon = 0.0)
    for j in 1:n
        xj = x[j]
        core, _ = ExaModels.add_con!(core, c_denom, i => -exp(xj) for i in 1:1)
    end
    y = [exp(x[j]) / denom[1] for j in 1:n]
    return (core, y), MathOptAI.Formulation(p, Any[denom], Any[c_denom])
end
