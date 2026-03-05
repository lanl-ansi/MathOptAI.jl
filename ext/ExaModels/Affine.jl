# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.Affine,
    x,
)
    m, n = size(p.A)
    y = ExaModels.variable(core, m)
    b = ExaModels.parameter(core, p.b)
    # Base: y[i] - b[i] = 0, parallelized over i ∈ 1:m
    c1 = ExaModels.constraint(
        core,
        y[i] - b[i] for i in 1:m;
        lcon = 0.0,
        ucon = 0.0,
    )
    # Augment column-by-column: subtract A[i,j]*x[j] from constraint i.
    # Full expression: y[i] - b[i] - sum_j A[i,j]*x[j] = 0  ⟺  y[i] = b[i] + A*x
    # x[j] with a fixed integer j produces a fixed Var node — GPU-friendly.
    for j in 1:n
        A_col = ExaModels.parameter(core, p.A[:, j])
        xj = x[j]
        ExaModels.constraint!(core, c1, i => -A_col[i] * xj for i in 1:m)
    end
    return y, MathOptAI.Formulation(p, Any[y], Any[c1])
end

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.Affine},
    x,
)
    A, b = p.predictor.A, p.predictor.b
    y = [
        b[i] + sum(
            A[i, j] * x[j] for j in axes(A, 2) if !iszero(A[i, j]);
            init = zero(eltype(b)),
        ) for i in axes(A, 1)
    ]
    return y, MathOptAI.Formulation(p)
end
