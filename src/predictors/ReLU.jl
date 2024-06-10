# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    ReLU() <: AbstractPredictor

An [`AbstractPredictor`](@ref) that implements the ReLU constraint
\$y = \\max(0, x)\$ as a non-smooth nonlinear constraint.

## Example

```jldoctest
julia> using JuMP

julia> import MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = MathOptAI.ReLU()
MathOptAI.ReLU()

julia> y = MathOptAI.add_predictor(model, f, x)
2-element Vector{VariableRef}:
 omelette_ReLU[1]
 omelette_ReLU[2]

julia> print(model)
Feasibility
Subject to
 omelette_ReLU[1] - max(0.0, x[1]) = 0
 omelette_ReLU[2] - max(0.0, x[2]) = 0
 omelette_ReLU[1] ≥ 0
 omelette_ReLU[2] ≥ 0
```
"""
struct ReLU <: AbstractPredictor end

function add_predictor(model::JuMP.Model, predictor::ReLU, x::Vector)
    ub = last.(_get_variable_bounds.(x))
    y = JuMP.@variable(model, [1:length(x)], base_name = "omelette_ReLU")
    _set_bounds_if_finite.(y, 0.0, ub)
    JuMP.@constraint(model, y .== max.(0, x))
    return y
end

"""
    ReLUBigM(M::Float64) <: AbstractPredictor

An [`AbstractPredictor`](@ref) that implements the ReLU constraint
\$y = \\max(0, x)\$ via a big-M MIP reformulation.

## Example

```jldoctest
julia> using JuMP

julia> import MathOptAI

julia> model = Model();

julia> @variable(model, -3 <= x[i in 1:2] <= i);

julia> f = MathOptAI.ReLUBigM(100.0)
MathOptAI.ReLUBigM(100.0)

julia> y = MathOptAI.add_predictor(model, f, x)
2-element Vector{VariableRef}:
 omelette_ReLU[1]
 omelette_ReLU[2]

julia> print(model)
Feasibility
Subject to
 -x[1] + omelette_ReLU[1] ≥ 0
 -x[2] + omelette_ReLU[2] ≥ 0
 omelette_ReLU[1] - _[5] ≤ 0
 -x[1] + omelette_ReLU[1] + 3 _[5] ≤ 3
 omelette_ReLU[2] - 2 _[6] ≤ 0
 -x[2] + omelette_ReLU[2] + 3 _[6] ≤ 3
 x[1] ≥ -3
 x[2] ≥ -3
 omelette_ReLU[1] ≥ 0
 omelette_ReLU[2] ≥ 0
 x[1] ≤ 1
 x[2] ≤ 2
 omelette_ReLU[1] ≤ 1
 omelette_ReLU[2] ≤ 2
 _[5] binary
 _[6] binary
```
"""
struct ReLUBigM <: AbstractPredictor
    M::Float64
end

function add_predictor(model::JuMP.Model, predictor::ReLUBigM, x::Vector)
    m = length(x)
    bounds = _get_variable_bounds.(x)
    y = JuMP.@variable(model, [1:m], base_name = "omelette_ReLU")
    _set_bounds_if_finite.(y, 0.0, last.(bounds))
    for i in 1:m
        lb, ub = bounds[i]
        z = JuMP.@variable(model, binary = true)
        JuMP.@constraint(model, y[i] >= x[i])
        U = min(ub, predictor.M)
        JuMP.@constraint(model, y[i] <= U * z)
        L = min(max(0.0, -lb), predictor.M)
        JuMP.@constraint(model, y[i] <= x[i] + L * (1 - z))
    end
    return y
end

"""
    ReLUSOS1() <: AbstractPredictor

An [`AbstractPredictor`](@ref) that implements the ReLU constraint
\$y = \\max(0, x)\$ by the reformulation:
```math
\\begin{aligned}
x = y - z \\\\
[y, z] \\in SOS1 \\\\
y, z \\ge 0
\\end{aligned}
```

## Example

```jldoctest
julia> using JuMP

julia> import MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2] >= -1);

julia> f = MathOptAI.ReLUSOS1()
MathOptAI.ReLUSOS1()

julia> y = MathOptAI.add_predictor(model, f, x)
2-element Vector{VariableRef}:
 omelette_ReLU[1]
 omelette_ReLU[2]

julia> print(model)
Feasibility
Subject to
 x[1] - omelette_ReLU[1] + _z[1] = 0
 x[2] - omelette_ReLU[2] + _z[2] = 0
 [omelette_ReLU[1], _z[1]] ∈ MathOptInterface.SOS1{Float64}([1.0, 2.0])
 [omelette_ReLU[2], _z[2]] ∈ MathOptInterface.SOS1{Float64}([1.0, 2.0])
 x[1] ≥ -1
 x[2] ≥ -1
 omelette_ReLU[1] ≥ 0
 omelette_ReLU[2] ≥ 0
 _z[1] ≥ 0
 _z[2] ≥ 0
 _z[1] ≤ 1
 _z[2] ≤ 1
```
"""
struct ReLUSOS1 <: AbstractPredictor end

function add_predictor(model::JuMP.Model, predictor::ReLUSOS1, x::Vector)
    m = length(x)
    bounds = _get_variable_bounds.(x)
    y = JuMP.@variable(model, [i in 1:m], base_name = "omelette_ReLU")
    _set_bounds_if_finite.(y, 0.0, last.(bounds))
    z = JuMP.@variable(model, [1:m], lower_bound = 0, base_name = "_z")
    _set_bounds_if_finite.(z, -Inf, -first.(bounds))
    JuMP.@constraint(model, x .== y - z)
    for i in 1:m
        JuMP.@constraint(model, [y[i], z[i]] in MOI.SOS1([1.0, 2.0]))
    end
    return y
end

"""
    ReLUQuadratic() <: AbstractPredictor

An [`AbstractPredictor`](@ref) that implements the ReLU constraint
\$y = \\max(0, x)\$ by the reformulation:
```math
\\begin{aligned}
x = y - z \\\\
y \\times z = 0 \\\\
y, z \\ge 0
\\end{aligned}
```

## Example

```jldoctest
julia> using JuMP

julia> import MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2] >= -1);

julia> f = MathOptAI.ReLUQuadratic()
MathOptAI.ReLUQuadratic()

julia> y = MathOptAI.add_predictor(model, f, x)
2-element Vector{VariableRef}:
 omelette_ReLU[1]
 omelette_ReLU[2]

julia> print(model)
Feasibility
Subject to
 x[1] - omelette_ReLU[1] + _z[1] = 0
 x[2] - omelette_ReLU[2] + _z[2] = 0
 omelette_ReLU[1]*_z[1] = 0
 omelette_ReLU[2]*_z[2] = 0
 x[1] ≥ -1
 x[2] ≥ -1
 omelette_ReLU[1] ≥ 0
 omelette_ReLU[2] ≥ 0
 _z[1] ≥ 0
 _z[2] ≥ 0
 _z[1] ≤ 1
 _z[2] ≤ 1
```
"""
struct ReLUQuadratic <: AbstractPredictor end

function add_predictor(model::JuMP.Model, predictor::ReLUQuadratic, x::Vector)
    m = length(x)
    bounds = _get_variable_bounds.(x)
    y = JuMP.@variable(model, [1:m], base_name = "omelette_ReLU")
    _set_bounds_if_finite.(y, 0.0, last.(bounds))
    z = JuMP.@variable(model, [1:m], base_name = "_z")
    _set_bounds_if_finite.(z, 0.0, -first.(bounds))
    JuMP.@constraint(model, x .== y - z)
    JuMP.@constraint(model, y .* z .== 0)
    return y
end
