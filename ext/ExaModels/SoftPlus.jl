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
    n = _length(x)
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
    p::MathOptAI.SoftPlus,
    x::AbstractVector,
)
    n = length(x)
    β = p.beta
    core, y = ExaModels.add_var(core, n; lvar = 0.0)
    cons = Any[]
    for i in 1:n
        core, c = ExaModels.add_con(
            core,
            y[i] - log(1 + exp(β * x[i])) / β;
            lcon = 0.0,
            ucon = 0.0,
        )
        push!(cons, c)
    end
    return (core, y), MathOptAI.Formulation(p, Any[y], cons)
end

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.SoftPlus},
    x,
)
    β = p.predictor.beta
    y = [log(1 + exp(β * x[i])) / β for i in 1:_length(x)]
    return (core, y), MathOptAI.Formulation(p)
end
