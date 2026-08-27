# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore{T},
    p::MathOptAI.ReducedSpace{MathOptAI.Permutation},
    x::Union{ExaModels.Variable,ExaModels.Expression},
) where {T}
    # The following usage of add_expr is broken:
    # https://github.com/madsuite-org/ExaModels.jl/issues/293
    # core, y = ExaModels.add_expr(core, x[i] for i in p.predictor.p)
    #
    # Instead, use a permutation matrix.
    P = zeros(T, x.length, x.length)
    for (i, p) in enumerate(p.predictor.p)
        P[i, p] = 1.0
    end
    p2 = MathOptAI.ReducedSpace(MathOptAI.Affine(P))
    (core, y), _ = MathOptAI.add_predictor(core, p2, x)
    return (core, y), MathOptAI.Formulation(p)
end
