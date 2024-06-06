# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    ReLUBigM(dimension::Int, M::Float64)

Represents the rectified linear unit relationship:
```math
f(x) = max.(0, x)
```

## Example

```jldoctest
julia> using JuMP, Omelette

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = Omelette.ReLUBigM(2, 100.0)
Omelette.ReLUBigM(2, 100.0)

julia> y = Omelette.add_predictor(model, f, x)
2-element Vector{VariableRef}:
 omelette_y[1]
 omelette_y[2]

julia> print(model)
Feasibility
Subject to
 omelette_y[1] ≥ 0
 omelette_y[2] ≥ 0
 -x[1] + omelette_y[1] ≥ 0
 -x[2] + omelette_y[2] ≥ 0
 omelette_y[1] - 100 _[5] ≤ 0
 omelette_y[2] - 100 _[6] ≤ 0
 -x[1] + omelette_y[1] + 100 _[5] ≤ 100
 -x[2] + omelette_y[2] + 100 _[6] ≤ 100
 _[5] binary
 _[6] binary
```
"""
struct ReLUBigM <: AbstractPredictor
    dimension::Int
    M::Float64
end

Base.size(x::ReLUBigM) = (x.dimension, x.dimension)

function _add_predictor_inner(
    model::JuMP.Model,
    predictor::ReLUBigM,
    x::Vector{JuMP.VariableRef},
    y::Vector{JuMP.VariableRef},
)
    z = JuMP.@variable(model, [1:length(x)], Bin)
    JuMP.@constraint(model, y .>= 0)
    JuMP.@constraint(model, y .>= x)
    JuMP.@constraint(model, y .<= predictor.M * z)
    JuMP.@constraint(model, y .<= x .+ predictor.M * (1 .- z))
    return
end

"""
    ReLUSOS1()

Implements the ReLU constraint \$y = max(0, x)\$ by the reformulation:
```math
\\begin{aligned}
x = x^+ - x^- \\\\
[x^+ , x^-] \\in SOS1
\\end{aligned}
```

## Example

```jldoctest
julia> using JuMP, Omelette

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = Omelette.ReLUSOS1(2)
Omelette.ReLUSOS1(2)

julia> y = Omelette.add_predictor(model, f, x)
2-element Vector{VariableRef}:
 omelette_y[1]
 omelette_y[2]

julia> print(model)
Feasibility
Subject to
 x[1] - omelette_y[1] + _[5] = 0
 x[2] - omelette_y[2] + _[6] = 0
 omelette_y[1] ≥ 0
 omelette_y[2] ≥ 0
 [omelette_y[1], _[5]] ∈ MathOptInterface.SOS1{Float64}([1.0, 2.0])
 [omelette_y[2], _[6]] ∈ MathOptInterface.SOS1{Float64}([1.0, 2.0])
 _[5] ≥ 0
 _[6] ≥ 0
```
"""
struct ReLUSOS1 <: AbstractPredictor
    dimension::Int
end

Base.size(x::ReLUSOS1) = (x.dimension, x.dimension)

function _add_predictor_inner(
    model::JuMP.Model,
    predictor::ReLUSOS1,
    x::Vector{JuMP.VariableRef},
    y::Vector{JuMP.VariableRef},
)
    for i in 1:length(x)
        z = JuMP.@variable(model, lower_bound = 0)
        JuMP.@constraint(model, y[i] >= 0)
        JuMP.@constraint(model, x[i] == y[i] - z)
        JuMP.@constraint(model, [y[i], z] in MOI.SOS1([1.0, 2.0]))
    end
    return
end

"""
    ReLUQuadratic()

Implements the ReLU constraint \$y = max(0, x)\$ by the reformulation:
```math
\\begin{aligned}
x = y - z \\\\
y \\times z = 0 \\\\
y, z \\ge 0
\\end{aligned}
```

## Example

```jldoctest
julia> using JuMP, Omelette

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = Omelette.ReLUQuadratic(2)
Omelette.ReLUQuadratic(2)

julia> y = Omelette.add_predictor(model, f, x)
2-element Vector{VariableRef}:
 omelette_y[1]
 omelette_y[2]

julia> print(model)
Feasibility
Subject to
 x[1] - omelette_y[1] + _z[1] = 0
 x[2] - omelette_y[2] + _z[2] = 0
 omelette_y[1] ≥ 0
 omelette_y[2] ≥ 0
 omelette_y[1]*_z[1] = 0
 omelette_y[2]*_z[2] = 0
 _z[1] ≥ 0
 _z[2] ≥ 0
```
"""
struct ReLUQuadratic <: AbstractPredictor
    dimension::Int
end

Base.size(x::ReLUQuadratic) = (x.dimension, x.dimension)

function _add_predictor_inner(
    model::JuMP.Model,
    predictor::ReLUQuadratic,
    x::Vector{JuMP.VariableRef},
    y::Vector{JuMP.VariableRef},
)
    for i in 1:length(x)
        z = JuMP.@variable(model, lower_bound = 0, base_name = "_z[$i]")
        JuMP.@constraint(model, y[i] >= 0)
        JuMP.@constraint(model, x[i] == y[i] - z)
        JuMP.@constraint(model, y[i] * z == 0)
    end
    return
end
