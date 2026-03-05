# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ReLU,
    x::ExaModels.AbstractVariable,
)
    n = _length(x)
    y = ExaModels.variable(core, n; lvar = 0.0)
    c1 = ExaModels.constraint(
        core,
        y[i] - max(0, x[i]) for i in 1:n;
        lcon = 0.0,
        ucon = 0.0,
    )
    return y, MathOptAI.Formulation(p, Any[y], Any[c1])
end

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ReLU,
    x::AbstractVector,
)
    n = length(x)
    y = ExaModels.variable(core, n; lvar = 0.0)
    cons = Any[
        ExaModels.constraint(core, y[i] - max(0, x[i]); lcon = 0.0, ucon = 0.0) for i in 1:n
    ]
    return y, MathOptAI.Formulation(p, Any[y], cons)
end

function MathOptAI.add_predictor(
    ::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.ReLU},
    x,
)
    y = [max(0, x[i]) for i in 1:_length(x)]
    return y, MathOptAI.Formulation(p)
end
