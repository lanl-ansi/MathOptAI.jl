# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.SoftMax,
    x::Union{ExaModels.Variable,ExaModels.Expression},
)
    n = x.length
    core, denom = ExaModels.add_var(core, 1; lvar = 0.0)
    core, y = ExaModels.add_var(core, n; lvar = 0.0, uvar = 1.0)
    # denom[1] - sum_j exp(x[j]) = 0
    core, c_denom =
        ExaModels.add_con(core, denom[1] for i in 1:1; lcon = 0.0, ucon = 0.0)
    core, _ = ExaModels.add_con!(core, c_denom, 1 => -exp(x[j]) for j in 1:n)
    # y[i] - exp(x[i]) / denom[1] = 0
    d = denom[1]
    core, c_y = ExaModels.add_con(
        core,
        y[i] - exp(x[i]) / d for i in 1:n;
        lcon = 0.0,
        ucon = 0.0,
    )
    return (core, y), MathOptAI.Formulation(p, [denom, y], Any[c_denom, c_y])
end
