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
  ├ moai_z[1] binary
  ├ -x[1] + moai_ReLU[1] ≥ 0
  ├ moai_ReLU[1] - moai_z[1] ≤ 0
  ├ -x[1] + moai_ReLU[1] + 3 moai_z[1] ≤ 3
  ├ moai_ReLU[2] ≥ 0
  ├ moai_ReLU[2] ≤ 2
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
    y = add_variables(model, x, m, "moai_ReLU")
    z = add_variables(model, x, m, "moai_z")
    JuMP.set_binary.(z)
    cons = Any[]
    for i in 1:m
        l, u = get_variable_bounds(x[i])
        lb = coalesce(max(0, l), 0)
        set_variable_bounds(cons, y[i], lb, max(0, u); optional = false)
        push!(cons, JuMP.BinaryRef(z[i]))
        push!(cons, JuMP.@constraint(model, y[i] >= x[i]))
        U = coalesce(min(predictor.M, u), predictor.M)
        push!(cons, JuMP.@constraint(model, y[i] <= U * z[i]))
        L = coalesce(min(predictor.M, max(-l, 0)), predictor.M)
        push!(cons, JuMP.@constraint(model, y[i] <= x[i] + L * (1 - z[i])))
    end
    return y, Formulation(predictor, Any[y; z], cons)
end
