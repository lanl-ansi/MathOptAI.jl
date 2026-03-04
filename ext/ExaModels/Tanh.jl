# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.Tanh,
    x::ExaModels.AbstractVariable,
)
    n = _exa_length(x)
    y = ExaModels.variable(core, n; lvar = -1.0, uvar = 1.0)
    c1 = ExaModels.constraint(
        core,
        y[i] - tanh(x[i]) for i in 1:n;
        lcon = 0.0,
        ucon = 0.0,
    )
    return y, MathOptAI.Formulation(p, [y], Any[c1])
end

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.Tanh,
    x::AbstractVector,
)
    n = length(x)
    y = ExaModels.variable(core, n; lvar = -1.0, uvar = 1.0)
    cons = [
        ExaModels.constraint(core, y[i] - tanh(x[i]); lcon = 0.0, ucon = 0.0) for i in 1:n
    ]
    return y, MathOptAI.Formulation(p, [y], cons)
end

function MathOptAI.add_predictor(
    ::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.Tanh},
    x,
)
    y = [tanh(x[i]) for i in 1:_exa_length(x)]
    return y, MathOptAI.Formulation(p)
end
