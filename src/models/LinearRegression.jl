# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    LinearRegression(parameters::Matrix)

Represents the linear relationship:
```math
f(x) = A x
```
where \$A\$ is the \$m \\times n\$ matrix `parameters`.

## Example

```jldoctest
julia> using JuMP, Omelette

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = Omelette.LinearRegression([2.0, 3.0])
Omelette.LinearRegression([2.0 3.0])

julia> y = Omelette.add_predictor(model, f, x)
1-element Vector{VariableRef}:
 omelette_y[1]

julia> print(model)
 Feasibility
 Subject to
  2 x[1] + 3 x[2] - omelette_y[1] = 0
```
"""
struct LinearRegression <: AbstractPredictor
    parameters::Matrix{Float64}
end

function LinearRegression(parameters::Vector{Float64})
    return LinearRegression(reshape(parameters, 1, length(parameters)))
end

Base.size(f::LinearRegression) = size(f.parameters)

function _add_predictor_inner(
    model::JuMP.Model,
    predictor::LinearRegression,
    x::Vector{JuMP.VariableRef},
    y::Vector{JuMP.VariableRef},
)
    JuMP.@constraint(model, predictor.parameters * x .== y)
    return
end
