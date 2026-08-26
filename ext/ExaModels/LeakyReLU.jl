# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.LeakyReLU,
    x::Union{ExaModels.Variable,ExaModels.Expression},
)
    (core, y_relu), f_relu = MathOptAI.add_predictor(core, p.relu, x)
    n = _length(x)
    η = p.negative_slope
    core, y = ExaModels.add_var(core, n)
    core, cons = ExaModels.add_con(
        core,
        y[i] - η * x[i] - (1 - η) * y_relu[i] for i in 1:n;
        lcon = 0.0,
        ucon = 0.0,
    )
    form = MathOptAI.Formulation(
        p,
        [f_relu.variables; y],
        [f_relu.constraints; cons],
    )
    return (core, y), form
end

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.LeakyReLU},
    x::Union{ExaModels.Variable,ExaModels.Expression},
)
    inner = MathOptAI.ReducedSpace(p.predictor.relu)
    (core, y_relu), _ = MathOptAI.add_predictor(core, inner, x)
    η = p.predictor.negative_slope
    core, y = ExaModels.add_expr(
        core,
        η * x[i] + (1 - η) * y_relu[i] for i in 1:_length(x)
    )
    return (core, y), MathOptAI.Formulation(p)
end
