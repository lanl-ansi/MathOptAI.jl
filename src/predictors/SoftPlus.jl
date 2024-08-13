# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    SoftPlus() <: AbstractPredictor

An [`AbstractPredictor`](@ref) that implements the SoftPlus constraint
\$y = \\log(1 + e^x)\$ as a smooth nonlinear constraint.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> y = MathOptAI.add_predictor(model, MathOptAI.SoftPlus(), x)
2-element Vector{VariableRef}:
 moai_SoftPlus[1]
 moai_SoftPlus[2]

julia> print(model)
Feasibility
Subject to
 moai_SoftPlus[1] - log(1.0 + exp(x[1])) = 0
 moai_SoftPlus[2] - log(1.0 + exp(x[2])) = 0
 moai_SoftPlus[1] ≥ 0
 moai_SoftPlus[2] ≥ 0
```
"""
struct SoftPlus <: AbstractPredictor end

function add_predictor(
    model::JuMP.Model,
    ::SoftPlus,
    x::Vector;
    reduced_space::Bool = false,
    kwargs...,
)
    if reduced_space
        return log.(1 .+ exp.(x))
    end
    y = JuMP.@variable(model, [1:length(x)], base_name = "moai_SoftPlus")
    _set_bounds_if_finite.(y, 0.0, Inf)
    JuMP.@constraint(model, y .== log.(1 .+ exp.(x)))
    return y
end
