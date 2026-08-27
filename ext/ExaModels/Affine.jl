# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.Affine,
    x::Union{ExaModels.Variable,ExaModels.Expression},
)
    m, n = size(p.A)
    core, y = ExaModels.add_var(core, m)
    core, c1 =
        ExaModels.add_con(core, y[i] for i in 1:m; lcon = p.b, ucon = p.b)
    IJV = [(i, j, -p.A[i, j]) for i in 1:m, j in 1:n]
    core, _ = ExaModels.add_con!(core, c1, i => v * x[j] for (i, j, v) in IJV)
    return (core, y), MathOptAI.Formulation(p, Any[y], Any[c1])
end

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.Affine},
    x::Union{ExaModels.Variable,ExaModels.Expression},
)
    m, n = size(p.predictor.A)
    core, A = ExaModels.add_par(core, p.predictor.A)
    core, b = ExaModels.add_par(core, p.predictor.b)
    core, y = ExaModels.add_expr(
        core,
        sum(A[i, j] * x[j] for j in 1:n) + b[i] for i in 1:m
    )
    return (core, y), MathOptAI.Formulation(p)
end
