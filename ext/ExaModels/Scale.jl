# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

function MathOptAI.add_predictor(core::ExaModels.ExaCore, p::MathOptAI.Scale, x)
    n = _length(x)
    core, s_param = ExaModels.add_par(core, p.scale)
    core, b_param = ExaModels.add_par(core, p.bias)
    core, y = ExaModels.add_var(core, n)
    core, c1 = ExaModels.add_con(
        core,
        y[i] - s_param[i] * x[i] - b_param[i] for i in 1:n;
        lcon = 0.0,
        ucon = 0.0,
    )
    return (core, y), MathOptAI.Formulation(p, Any[y], Any[c1])
end

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.Scale},
    x,
)
    s, b = p.predictor.scale, p.predictor.bias
    y = [s[i] * x[i] + b[i] for i in 1:_length(x)]
    return (core, y), MathOptAI.Formulation(p)
end
