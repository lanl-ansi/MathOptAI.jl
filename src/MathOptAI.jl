# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module MathOptAI

import Distributions
import JuMP
import MathOptInterface as MOI

"""
    abstract type AbstractPredictor end

An abstract type representig different types of prediction models.

## Methods

All subtypes must implement:

 * `add_predictor`
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
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = MathOptAI.Affine([2.0, 3.0])
MathOptAI.Affine([2.0 3.0], [0.0])

julia> y = MathOptAI.add_predictor(model, f, x)
1-element Vector{VariableRef}:
 omelette_Affine[1]

julia> print(model)
Feasibility
Subject to
 2 x[1] + 3 x[2] - omelette_Affine[1] = 0
```
"""
function add_predictor end

include("utilities.jl")

for dir in ("predictors", "constraints")
    for file in filter(
        x -> endswith(x, ".jl"),
        readdir(joinpath(@__DIR__, dir); join = true),
    )
        include(file)
    end
end

end  # module MathOptAI
