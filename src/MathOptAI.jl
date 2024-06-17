# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAI

import Distributions
import JuMP
import MathOptInterface as MOI

"""
    abstract type AbstractPredictor end

An abstract type representing different types of prediction models.

## Methods

All subtypes must implement:

 * [`add_predictor`](@ref)
"""
abstract type AbstractPredictor end

"""
    add_predictor(
        model::JuMP.Model,
        predictor::AbstractPredictor,
        x::Vector,
    )::Vector{JuMP.VariableRef}

Return a `Vector{JuMP.VariableRef}` representing `y` such that
`y = predictor(x)`.

## Example

```jldoctest
julia> using JuMP

julia> import MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = MathOptAI.Affine([2.0, 3.0])
MathOptAI.Affine([2.0 3.0], [0.0])

julia> y = MathOptAI.add_predictor(model, f, x)
1-element Vector{VariableRef}:
 moai_Affine[1]

julia> print(model)
Feasibility
Subject to
 2 x[1] + 3 x[2] - moai_Affine[1] = 0
```
"""
function add_predictor end

"""
    add_predictor(
        model::JuMP.Model,
        predictor::AbstractPredictor,
        x::Matrix,
    )::Matrix{JuMP.VariableRef}

Return a `Matrix{JuMP.VariableRef}`, representing `y` such that
`y[:, i] = predictor(x[:, i])` for each columnn `i`.

## Example

```jldoctest
julia> using JuMP

julia> import MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2, 1:3]);

julia> f = MathOptAI.Affine([2.0, 3.0])
MathOptAI.Affine([2.0 3.0], [0.0])

julia> y = MathOptAI.add_predictor(model, f, x)
1×3 Matrix{VariableRef}:
 moai_Affine[1]  moai_Affine[1]  moai_Affine[1]

julia> print(model)
Feasibility
Subject to
 2 x[1,1] + 3 x[2,1] - moai_Affine[1] = 0
 2 x[1,2] + 3 x[2,2] - moai_Affine[1] = 0
 2 x[1,3] + 3 x[2,3] - moai_Affine[1] = 0
```
"""
function add_predictor(
    model::JuMP.Model,
    predictor,
    x::Matrix,
)::Matrix{JuMP.VariableRef}
    y = map(j -> add_predictor(model, predictor, x[:, j]), 1:size(x, 2))
    return reduce(hcat, y)
end

include("utilities.jl")

for file in filter(
    x -> endswith(x, ".jl"),
    readdir(joinpath(@__DIR__, "predictors"); join = true),
)
    include(file)
end

for sym in names(@__MODULE__; all = true)
    if !Base.isidentifier(sym) || sym in (:eval, :include)
        continue
    elseif startswith("$sym", "_")
        continue
    end
    @eval export $sym
end

end  # module MathOptAI
