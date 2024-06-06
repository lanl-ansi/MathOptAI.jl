# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    LogisticRegression(parameters::Matrix)

Represents the linear relationship:
```math
f(x) = \\frac{1}{1 + e^{-A x}}
```
where \$A\$ is the \$m \\times n\$ matrix `parameters`.

## Example

```jldoctest
julia> using JuMP, Omelette

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = Omelette.LogisticRegression([2.0, 3.0])
Omelette.LogisticRegression([2.0 3.0])

julia> y = Omelette.add_predictor(model, f, x)
1-element Vector{VariableRef}:
 omelette_y[1]

julia> print(model)
Feasibility
Subject to
 (1.0 / (1.0 + exp(-2 x[1] - 3 x[2]))) - omelette_y[1] = 0
```
"""
struct LogisticRegression <: AbstractPredictor
    parameters::Matrix{Float64}
end

function LogisticRegression(parameters::Vector{Float64})
    return LogisticRegression(reshape(parameters, 1, length(parameters)))
end

function add_predictor(
    model::JuMP.Model,
    predictor::LogisticRegression,
    x::Vector{JuMP.VariableRef},
)
    m = size(predictor.parameters, 1)
    y = JuMP.@variable(model, [1:m], base_name = "omelette_y")
    JuMP.@constraint(model, 1 ./ (1 .+ exp.(-predictor.parameters * x)) .== y)
    return y
end
