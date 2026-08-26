# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.Scale,
    x::Union{ExaModels.Variable,ExaModels.Expression},
)
    n = _length(x)
    core, y = ExaModels.add_var(core, n)
    core, c1 = ExaModels.add_con(
        core,
        y[i] - si * x[i] for (i, si) in enumerate(p.scale);
        lcon = p.bias,
        ucon = p.bias,
    )
    return (core, y), MathOptAI.Formulation(p, Any[y], Any[c1])
end

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.Scale},
    x::Union{ExaModels.Variable,ExaModels.Expression},
)
    iter = enumerate(zip(p.predictor.scale, p.predictor.bias))
    core, y = ExaModels.add_expr(core, si * x[i] + bi for (i, (si, bi)) in iter)
    return (core, y), MathOptAI.Formulation(p)
end
