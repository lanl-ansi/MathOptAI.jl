# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    ReLUBigM(M::Float64) <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents the relationship:
```math
y = \\max\\{0, x\\}
```
via the big-M MIP reformulation:
```math
\\begin{aligned}
y \\ge 0            \\\\
y \\ge x            \\\\
y \\le M z          \\\\
y \\le x + M(1 - z) \\\\
z \\in\\{0, 1\\}
\\end{aligned}
```

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
│ ├ moai_z[1]
│ └ moai_z[2]
└ constraints [12]
  ├ moai_ReLU[1] ≥ 0
  ├ moai_ReLU[1] ≤ 1
  ├ moai_ReLU[2] ≥ 0
  ├ moai_ReLU[2] ≤ 2
  ├ moai_z[1] binary
  ├ -x[1] + moai_ReLU[1] ≥ 0
  ├ moai_ReLU[1] - moai_z[1] ≤ 0
  ├ -x[1] + moai_ReLU[1] + 3 moai_z[1] ≤ 3
  ├ moai_z[2] binary
  ├ -x[2] + moai_ReLU[2] ≥ 0
  ├ moai_ReLU[2] - 2 moai_z[2] ≤ 0
  └ -x[2] + moai_ReLU[2] + 3 moai_z[2] ≤ 3
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
    y = add_variables(model, predictor, x, m, "moai_ReLU")
    cons = _set_direct_bounds(x -> max(0, x), 0, nothing, x, y)
    formulation = Formulation(predictor, Any[], cons)
    append!(formulation.variables, y)
    z = add_variables(model, predictor, x, m, "moai_z")
    JuMP.set_binary.(z)
    append!(formulation.variables, z)
    append!(formulation.constraints, JuMP.BinaryRef.(z))
    for i in 1:m
        lb, ub = _get_variable_bounds(x[i])
        c = JuMP.@constraint(model, y[i] >= x[i])
        push!(formulation.constraints, c)
        c = JuMP.@constraint(model, y[i] <= min(ub, predictor.M) * z[i])
        push!(formulation.constraints, c)
        L = min(max(0.0, -lb), predictor.M)
        c = JuMP.@constraint(model, y[i] <= x[i] + L * (1 - z[i]))
        push!(formulation.constraints, c)
    end
    return y, formulation
end
