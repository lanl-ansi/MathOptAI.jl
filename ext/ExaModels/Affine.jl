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
    core, y = ExaModels.add_var(core, m)
    core, b = ExaModels.add_par(core, p.b)
    core, A = ExaModels.add_par(core, p.A)
    core, c1 = ExaModels.add_con(
        core,
        y[i] - b[i] for i in 1:m;
        lcon = 0.0,
        ucon = 0.0,
    )
    core, _ = ExaModels.add_con!(
        core,
        c1,
        i => -A[j, i] * x[j] for i in 1:m, j in 1:n
    )
    return (core, y), MathOptAI.Formulation(p, Any[y], Any[c1])
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
    return (core, y), MathOptAI.Formulation(p)
end
