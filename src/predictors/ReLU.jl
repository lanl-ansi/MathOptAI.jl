# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    ReLU() <: AbstractPredictor

An [`AbstractPredictor`](@ref) that implements the ReLU constraint
\$y = \\max\\{0, x\\}\$ as a non-smooth nonlinear constraint.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = MathOptAI.ReLU()
ReLU()

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
2-element Vector{VariableRef}:
 moai_ReLU[1]
 moai_ReLU[2]

julia> formulation
ReLU()
├ variables [2]
│ ├ moai_ReLU[1]
│ └ moai_ReLU[2]
└ constraints [4]
  ├ moai_ReLU[1] ≥ 0
  ├ moai_ReLU[2] ≥ 0
  ├ moai_ReLU[1] - max(0.0, x[1]) = 0
  └ moai_ReLU[2] - max(0.0, x[2]) = 0

julia> y, formulation =
           MathOptAI.add_predictor(model, MathOptAI.ReducedSpace(f), x);

julia> y
2-element Vector{NonlinearExpr}:
 max(0.0, x[1])
 max(0.0, x[2])

julia> formulation
ReducedSpace(ReLU())
├ variables [0]
└ constraints [0]
```
"""
struct ReLU <: AbstractPredictor end

function add_predictor(model::JuMP.AbstractModel, predictor::ReLU, x::Vector)
    ub = last.(_get_variable_bounds.(x))
    y = JuMP.@variable(model, [1:length(x)], base_name = "moai_ReLU")
    _set_bounds_if_finite.(y, 0, max.(0, ub))
    cons = JuMP.@constraint(model, y .== max.(0, x))
    constraints = Any[JuMP.LowerBoundRef.(y); cons]
    return y, Formulation(predictor, y, constraints)
end

function add_predictor(
    ::JuMP.AbstractModel,
    predictor::ReducedSpace{ReLU},
    x::Vector,
)
    return max.(0, x), Formulation(predictor)
end

"""
    ReLUBigM(M::Float64) <: AbstractPredictor

An [`AbstractPredictor`](@ref) that implements the ReLU constraint
\$y = \\max\\{0, x\\}\$ via a big-M MIP reformulation.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, -3 <= x[i in 1:2] <= i);

julia> f = MathOptAI.ReLUBigM(100.0)
ReLUBigM(100.0)

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
2-element Vector{VariableRef}:
 moai_ReLU[1]
 moai_ReLU[2]

julia> formulation
ReLUBigM(100.0)
├ variables [4]
│ ├ moai_ReLU[1]
│ ├ moai_ReLU[2]
│ ├ _[5]
│ └ _[6]
└ constraints [8]
  ├ _[5] binary
  ├ -x[1] + moai_ReLU[1] ≥ 0
  ├ moai_ReLU[1] - _[5] ≤ 0
  ├ -x[1] + moai_ReLU[1] + 3 _[5] ≤ 3
  ├ _[6] binary
  ├ -x[2] + moai_ReLU[2] ≥ 0
  ├ moai_ReLU[2] - 2 _[6] ≤ 0
  └ -x[2] + moai_ReLU[2] + 3 _[6] ≤ 3
```
"""
struct ReLUBigM <: AbstractPredictor
    M::Float64
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReLUBigM,
    x::Vector,
)
    m = length(x)
    bounds = _get_variable_bounds.(x)
    y = JuMP.@variable(model, [1:m], base_name = "moai_ReLU")
    _set_bounds_if_finite.(y, 0, max.(0, last.(bounds)))
    formulation = Formulation(predictor)
    append!(formulation.variables, y)
    for i in 1:m
        lb, ub = bounds[i]
        z = JuMP.@variable(model, binary = true)
        push!(formulation.variables, z)
        push!(formulation.constraints, JuMP.BinaryRef(z))
        c = JuMP.@constraint(model, y[i] >= x[i])
        push!(formulation.constraints, c)
        c = JuMP.@constraint(model, y[i] <= min(ub, predictor.M) * z)
        push!(formulation.constraints, c)
        L = min(max(0.0, -lb), predictor.M)
        c = JuMP.@constraint(model, y[i] <= x[i] + L * (1 - z))
        push!(formulation.constraints, c)
    end
    return y, formulation
end

"""
    ReLUSOS1() <: AbstractPredictor

An [`AbstractPredictor`](@ref) that implements the ReLU constraint
\$y = \\max\\{0, x\\}\$ by the reformulation:
```math
\\begin{aligned}
x = y - z \\\\
[y, z] \\in SOS1 \\\\
y, z \\ge 0
\\end{aligned}
```

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2] >= -1);

julia> f = MathOptAI.ReLUSOS1()
ReLUSOS1()

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
2-element Vector{VariableRef}:
 moai_ReLU[1]
 moai_ReLU[2]

julia> formulation
ReLUSOS1()
├ variables [4]
│ ├ moai_ReLU[1]
│ ├ moai_ReLU[2]
│ ├ moai_z[1]
│ └ moai_z[2]
└ constraints [4]
  ├ x[1] - moai_ReLU[1] + moai_z[1] = 0
  ├ x[2] - moai_ReLU[2] + moai_z[2] = 0
  ├ [moai_ReLU[1], moai_z[1]] ∈ MathOptInterface.SOS1{Float64}([1.0, 2.0])
  └ [moai_ReLU[2], moai_z[2]] ∈ MathOptInterface.SOS1{Float64}([1.0, 2.0])
```
"""
struct ReLUSOS1 <: AbstractPredictor end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReLUSOS1,
    x::Vector,
)
    m = length(x)
    bounds = _get_variable_bounds.(x)
    y = JuMP.@variable(model, [i in 1:m], base_name = "moai_ReLU")
    _set_bounds_if_finite.(y, 0, max.(0, last.(bounds)))
    z = JuMP.@variable(model, [1:m], lower_bound = 0, base_name = "moai_z")
    _set_bounds_if_finite.(z, nothing, -first.(bounds))
    cons = JuMP.@constraint(model, x .== y - z)
    formulation = Formulation(predictor, Any[y; z], Any[cons;])
    for i in 1:m
        c = JuMP.@constraint(model, [y[i], z[i]] in MOI.SOS1([1.0, 2.0]))
        push!(formulation.constraints, c)
    end
    return y, formulation
end

"""
    ReLUQuadratic() <: AbstractPredictor

An [`AbstractPredictor`](@ref) that implements the ReLU constraint
\$y = \\max\\{0, x\\}\$ by the reformulation:
```math
\\begin{aligned}
x = y - z \\\\
y \\times z = 0 \\\\
y, z \\ge 0
\\end{aligned}
```

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2] >= -1);

julia> f = MathOptAI.ReLUQuadratic()
ReLUQuadratic()

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
2-element Vector{VariableRef}:
 moai_ReLU[1]
 moai_ReLU[2]

julia> formulation
ReLUQuadratic()
├ variables [4]
│ ├ moai_ReLU[1]
│ ├ moai_ReLU[2]
│ ├ moai_z[1]
│ └ moai_z[2]
└ constraints [4]
  ├ x[1] - moai_ReLU[1] + moai_z[1] = 0
  ├ x[2] - moai_ReLU[2] + moai_z[2] = 0
  ├ moai_ReLU[1]*moai_z[1] = 0
  └ moai_ReLU[2]*moai_z[2] = 0
```
"""
struct ReLUQuadratic <: AbstractPredictor end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReLUQuadratic,
    x::Vector,
)
    m = length(x)
    bounds = _get_variable_bounds.(x)
    y = JuMP.@variable(model, [1:m], base_name = "moai_ReLU")
    _set_bounds_if_finite.(y, 0, max.(0, last.(bounds)))
    z = JuMP.@variable(model, [1:m], base_name = "moai_z")
    _set_bounds_if_finite.(z, 0, max.(0, -first.(bounds)))
    c1 = JuMP.@constraint(model, x .== y - z)
    c2 = JuMP.@constraint(model, y .* z .== 0)
    return y, Formulation(predictor, Any[y; z], Any[c1; c2])
end
