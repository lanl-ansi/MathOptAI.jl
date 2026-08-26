# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ReLUEpigraph,
    x::Union{ExaModels.Variable,ExaModels.Expression},
)
    n = _length(x)
    core, y = ExaModels.add_var(core, n; lvar = 0.0)
    core, c1 = ExaModels.add_con(
        core,
        y[i] - x[i] for i in 1:n;
        lcon = 0.0,
        ucon = Inf,
    )
    return (core, y), MathOptAI.Formulation(p, Any[y], Any[c1])
end
