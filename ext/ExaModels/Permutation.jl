# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

function MathOptAI.add_predictor(
    ::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{MathOptAI.Permutation},
    x,
)
    y = [x[p.predictor.p[i]] for i in eachindex(p.predictor.p)]
    return y, MathOptAI.Formulation(p)
end
