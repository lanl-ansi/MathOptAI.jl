# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAIExaModelsFluxExt

import ExaModels
import Flux
import MathOptAI

_length(x::Union{ExaModels.Variable,ExaModels.Expression}) = x.length

_length(x::AbstractVector) = length(x)

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.GrayBox{<:Flux.Chain},
    x::Any;
    kwargs...,
)
    J(x) = only(Flux.jacobian(p.predictor, Float32.(x)))
    core, y, oracle = ExaModels.embed_oracle(
        core,
        x,
        _length(x);
        f! = (ret, x) -> ret .= p.predictor(Float32.(x)),
        jvp! = (ret, x, v) -> ret .= J(x) * v,
        vjp! = (ret, x, w) -> ret .= J(x)' * w,
        # From the ExaModels docs: "Use `adapt=Val(true)` to have arrays
        # automatically copied to CPU before each callback invocation.
        adapt = Val(true),
    )
    return (core, y), MathOptAI.Formulation(p, Any[y], Any[oracle])
end

end  # MathOptAIExaModelsFluxExt
