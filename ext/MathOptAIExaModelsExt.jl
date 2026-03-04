# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAIExaModelsExt

import ExaModels
import MathOptAI

_exa_length(x::ExaModels.Variable) = x.length
_exa_length(x::ExaModels.Subexpr) = x.length
_exa_length(x::ExaModels.ReducedSubexpr) = x.length
_exa_length(x::AbstractVector) = length(x)

for file in filter(endswith(".jl"), readdir(joinpath(@__DIR__, "ExaModels")))
    include(joinpath(@__DIR__, "ExaModels", file))
end

end  # module MathOptAIExaModelsExt
