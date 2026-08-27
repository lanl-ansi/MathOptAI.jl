# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.SoftPlus,
    x::Union{ExaModels.Variable,ExaModels.Expression},
)
    n = x.length
    β = p.beta
    core, y = ExaModels.add_var(core, n; lvar = 0.0)
    core, c1 = ExaModels.add_con(
        core,
        y[i] - log(1 + exp(β * x[i])) / β for i in 1:n;
        lcon = 0.0,
        ucon = 0.0,
    )
    return (core, y), MathOptAI.Formulation(p, Any[y], Any[c1])
end

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.SoftPlus},
    x::Union{ExaModels.Variable,ExaModels.Expression},
)
    β = p.predictor.beta
    core, y =
        ExaModels.add_expr(core, log(1 + exp(β * x[i])) / β for i in 1:x.length)
    return (core, y), MathOptAI.Formulation(p)
end
