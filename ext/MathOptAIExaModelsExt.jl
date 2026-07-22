# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAIExaModelsExt

import ExaModels
import MathOptAI

# This is a bug with ExaModels.
using ExaModels: Constant

function _length(x::Union{ExaModels.Variable,ExaModels.Expression})
    return x.length
end

_length(x::AbstractVector) = length(x)

"""
    add_predictor(
        model::ExaModels.ExaCore,
        predictor::Any,
        x::Any;
        reduced_space::Bool = false,
        kwargs...,
    )::Tuple{<:Vector,<:AbstractFormulation}

Return a `Vector` representing `y` such that `y = predictor(x)` and an
[`AbstractFormulation`](@ref) containing the variables and constraints that were
added to the model.

The element type of `x` is deliberately unspecified.

## Keyword arguments

 * `reduced_space`: if `true`, wrap `predictor` in [`ReducedSpace`](@ref) before
   adding to the model.

All other keyword arguments are passed to [`build_predictor`](@ref).

## Example

```jldoctest
julia> using ExaModels, MathOptAI

julia> model = ExaModels.ExaCore(; concrete = Val(true))
An ExaCore

  Float type: ...................... Float64
  Array type: ...................... Vector{Float64}
  Backend: ......................... Nothing

  number of objective patterns: .... 0
  number of constraint patterns: ... 0


julia> core, x = ExaModels.add_var(core, 2)
Variable

  x ∈ R^{2}


julia> f = MathOptAI.Affine([2.0, 3.0])
Affine(A, b) [input: 2, output: 1]

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
Variable

  x ∈ R^{1}


julia> formulation
Affine(A, b) [input: 2, output: 1]
├ variables [1]
│ └ Variable

  x ∈ R^{1}

└ constraints [1]
  └ Constraint

  s.t. (...)
       g♭ ≤ [g(x,θ,p)]_{p ∈ P} ≤ g♯

  where |P| = 1
```
"""
function MathOptAI.add_predictor(
    model::ExaModels.ExaCore,
    predictor::Any,
    x::Any;
    reduced_space::Bool = false,
    kwargs...,
)
    inner = MathOptAI.build_predictor(predictor; kwargs...)
    return MathOptAI.add_predictor(model, inner, x; reduced_space)
end

function MathOptAI.add_predictor(
    model::ExaModels.ExaCore,
    predictor::MathOptAI.AbstractPredictor,
    x::Any;
    reduced_space::Bool = false,
)
    if reduced_space
        inner = MathOptAI.ReducedSpace(predictor)
        return MathOptAI.add_predictor(model, inner, x)
    end
    return MathOptAI.add_predictor(model, predictor, x)
end

include("ExaModels/Affine.jl")
include("ExaModels/GELU.jl")
include("ExaModels/LeakyReLU.jl")
include("ExaModels/Permutation.jl")
include("ExaModels/Pipeline.jl")
include("ExaModels/ReLU.jl")
include("ExaModels/ReLUEpigraph.jl")
include("ExaModels/Scale.jl")
include("ExaModels/Sigmoid.jl")
include("ExaModels/SoftMax.jl")
include("ExaModels/SoftPlus.jl")
include("ExaModels/Tanh.jl")

end  # module MathOptAIExaModelsExt
