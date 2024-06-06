# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    ReLUBigM(M::Float64)

Represents the rectified linear unit relationship:
```math
f(x) = max.(0, x)
```

## Example

```jldoctest
julia> using JuMP, Omelette

julia> model = Model();

julia> @variable(model, -3 <= x[i in 1:2] <= i);

julia> f = Omelette.ReLUBigM(100.0)
Omelette.ReLUBigM(100.0)

julia> y = Omelette.add_predictor(model, f, x)
2-element Vector{VariableRef}:
 omelette_y[1]
 omelette_y[2]

julia> print(model)
Feasibility
Subject to
 -x[1] + omelette_y[1] ≥ 0
 -x[2] + omelette_y[2] ≥ 0
 omelette_y[1] - _[5] ≤ 0
 -x[1] + omelette_y[1] + 3 _[5] ≤ 3
 omelette_y[2] - 2 _[6] ≤ 0
 -x[2] + omelette_y[2] + 3 _[6] ≤ 3
 x[1] ≥ -3
 x[2] ≥ -3
 omelette_y[1] ≥ 0
 omelette_y[2] ≥ 0
 x[1] ≤ 1
 x[2] ≤ 2
 omelette_y[1] ≤ 1
 omelette_y[2] ≤ 2
 _[5] binary
 _[6] binary
```
"""
struct ReLUBigM <: AbstractPredictor
    M::Float64
end

function add_predictor(
    model::JuMP.Model,
    predictor::ReLUBigM,
    x::Vector{JuMP.VariableRef},
)
    m = length(x)
    lb, ub = _get_variable_bounds(x)
    y = JuMP.@variable(model, [1:m], base_name = "omelette_y")
    _set_bounds_if_finite.(y, 0.0, ub)
    for i in 1:m
        z = JuMP.@variable(model, binary = true)
        JuMP.@constraint(model, y[i] >= x[i])
        U = min(ub[i], predictor.M)
        JuMP.@constraint(model, y[i] <= U * z)
        L = min(max(0.0, -lb[i]), predictor.M)
        JuMP.@constraint(model, y[i] <= x[i] + L * (1 - z))
    end
    return y
end

"""
    ReLUSOS1()

Implements the ReLU constraint \$y = max(0, x)\$ by the reformulation:
```math
\\begin{aligned}
x = y - z \\\\
[y, z] \\in SOS1 \\\\
y, z \\ge 0
\\end{aligned}
```

## Example

```jldoctest
julia> using JuMP, Omelette

julia> model = Model();

julia> @variable(model, x[1:2] >= -1);

julia> f = Omelette.ReLUSOS1()
Omelette.ReLUSOS1()

julia> y = Omelette.add_predictor(model, f, x)
2-element Vector{VariableRef}:
 omelette_y[1]
 omelette_y[2]

julia> print(model)
Feasibility
Subject to
 x[1] - omelette_y[1] + _z[1] = 0
 x[2] - omelette_y[2] + _z[2] = 0
 [omelette_y[1], _z[1]] ∈ MathOptInterface.SOS1{Float64}([1.0, 2.0])
 [omelette_y[2], _z[2]] ∈ MathOptInterface.SOS1{Float64}([1.0, 2.0])
 x[1] ≥ -1
 x[2] ≥ -1
 omelette_y[1] ≥ 0
 omelette_y[2] ≥ 0
 _z[1] ≥ 0
 _z[2] ≥ 0
 _z[1] ≤ 1
 _z[2] ≤ 1
```
"""
struct ReLUSOS1 <: AbstractPredictor end

function add_predictor(
    model::JuMP.Model,
    predictor::ReLUSOS1,
    x::Vector{JuMP.VariableRef},
)
    m = length(x)
    lb, ub = _get_variable_bounds(x)
    y = JuMP.@variable(model, [i in 1:m], base_name = "omelette_y")
    _set_bounds_if_finite.(y, 0.0, ub)
    z = JuMP.@variable(model, [1:m], lower_bound = 0, base_name = "_z")
    _set_bounds_if_finite.(z, -Inf, -lb)
    JuMP.@constraint(model, x .== y - z)
    for i in 1:m
        JuMP.@constraint(model, [y[i], z[i]] in MOI.SOS1([1.0, 2.0]))
    end
    return y
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
julia> model = Model();

julia> @variable(model, x[1:2] >= -1);

julia> f = Omelette.ReLUQuadratic()
Omelette.ReLUQuadratic()

julia> y = Omelette.add_predictor(model, f, x)
2-element Vector{VariableRef}:
 omelette_y[1]
 omelette_y[2]

julia> print(model)
Feasibility
Subject to
 x[1] - omelette_y[1] + _z[1] = 0
 x[2] - omelette_y[2] + _z[2] = 0
 omelette_y[1]*_z[1] = 0
 omelette_y[2]*_z[2] = 0
 x[1] ≥ -1
 x[2] ≥ -1
 omelette_y[1] ≥ 0
 omelette_y[2] ≥ 0
 _z[1] ≥ 0
 _z[2] ≥ 0
 _z[1] ≤ 1
 _z[2] ≤ 1
```
"""
struct ReLUQuadratic <: AbstractPredictor end

function add_predictor(
    model::JuMP.Model,
    predictor::ReLUQuadratic,
    x::Vector{JuMP.VariableRef},
)
    m = length(x)
    lb, ub = _get_variable_bounds(x)
    y = JuMP.@variable(model, [1:m], base_name = "omelette_y")
    _set_bounds_if_finite.(y, 0.0, ub)
    z = JuMP.@variable(model, [1:m], base_name = "_z")
    _set_bounds_if_finite.(z, 0.0, -lb)
    JuMP.@constraint(model, x .== y - z)
    JuMP.@constraint(model, y .* z .== 0)
    return y
end
