# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.SoftPlus,
    x::ExaModels.AbstractVariable,
)
    n = _exa_length(x)
    β = p.beta
    y = ExaModels.variable(core, n; lvar = 0.0)
    c1 = ExaModels.constraint(
        core,
        y[i] - log(1 + exp(β * x[i])) / β for i in 1:n;
        lcon = 0.0,
        ucon = 0.0,
    )
    return y, MathOptAI.Formulation(p, [y], Any[c1])
end

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.SoftPlus,
    x::AbstractVector,
)
    n = length(x)
    β = p.beta
    y = ExaModels.variable(core, n; lvar = 0.0)
    cons = [
        ExaModels.constraint(
            core,
            y[i] - log(1 + exp(β * x[i])) / β;
            lcon = 0.0,
            ucon = 0.0,
        ) for i in 1:n
    ]
    return y, MathOptAI.Formulation(p, [y], cons)
end

function MathOptAI.add_predictor(
    ::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.SoftPlus},
    x,
)
    β = p.predictor.beta
    y = [log(1 + exp(β * x[i])) / β for i in 1:_exa_length(x)]
    return y, MathOptAI.Formulation(p)
end
