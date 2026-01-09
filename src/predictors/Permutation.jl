# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    Permutation(p::Vector{Int}) <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents the relationship:
```math
y = x[p]
```

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = MathOptAI.Permutation([2, 1])
Permutation([2, 1])

julia> y, formulation =
           MathOptAI.add_predictor(model, f, x; reduced_space = true);

julia> y
2-element Vector{VariableRef}:
 x[2]
 x[1]
```
"""
struct Permutation <: AbstractPredictor
    p::Vector{Int}
end

output_size(::Permutation, input_size) = input_size

function add_predictor(
    ::JuMP.AbstractModel,
    predictor::ReducedSpace{Permutation},
    x::Vector,
)
    return x[predictor.predictor.p], Formulation(predictor)
end
