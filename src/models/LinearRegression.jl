# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    LinearRegression(
        A::Matrix{Float64},
        b::Vector{Float64} = zeros(size(A, 1)),
    )

Represents the linear relationship:
```math
f(x) = A x + b
```
where \$A\$ is the \$m \\times n\$ matrix `A`.

## Example

```jldoctest
julia> using JuMP, Omelette

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = Omelette.LinearRegression([2.0, 3.0])
Omelette.LinearRegression([2.0 3.0], [0.0])

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
    A::Matrix{Float64}
    b::Vector{Float64}
end

function LinearRegression(A::Matrix{Float64})
    return LinearRegression(A, zeros(size(A, 1)))
end

function LinearRegression(A::Vector{Float64})
    return LinearRegression(reshape(A, 1, length(A)), [0.0])
end

function add_predictor(
    model::JuMP.Model,
    predictor::LinearRegression,
    x::Vector{JuMP.VariableRef},
)
    m = size(predictor.A, 1)
    y = JuMP.@variable(model, [1:m], base_name = "omelette_y")
    JuMP.@constraint(model, predictor.A * x .+ predictor.b .== y)
    return y
end
