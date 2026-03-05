# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.Pipeline,
    x,
)
    form = MathOptAI.PipelineFormulation(p, Any[])
    for layer in p.layers
        x, inner = MathOptAI.add_predictor(core, layer, x)
        push!(form.layers, inner)
    end
    return x, form
end

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.Pipeline},
    x,
)
    form = MathOptAI.PipelineFormulation(p, Any[])
    for layer in p.predictor.layers
        x, inner =
            MathOptAI.add_predictor(core, MathOptAI.ReducedSpace(layer), x)
        push!(form.layers, inner)
    end
    return x, form
end
