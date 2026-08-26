# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{MathOptAI.Permutation},
    x::Union{ExaModels.Variable,ExaModels.Expression},
)
    core, y = ExaModels.add_expr(core, x[i] for i in p.predictor.p)
    return (core, y), MathOptAI.Formulation(p)
end
