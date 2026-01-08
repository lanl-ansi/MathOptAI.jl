# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    ReLUSOS1() <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents the relationship:
```math
y = \\max\\{0, x\\}
```
by the reformulation:
```math
\\begin{aligned}
x = y - z           \\\\
[y, z] \\in SOS1    \\\\
y, z \\ge 0
\\end{aligned}
```

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, -1 <= x[i in 1:2] <= i);

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
└ constraints [12]
  ├ moai_ReLU[1] ≥ 0
  ├ moai_ReLU[1] ≤ 1
  ├ moai_z[1] ≥ 0
  ├ moai_z[1] ≤ 1
  ├ x[1] - moai_ReLU[1] + moai_z[1] = 0
  ├ [moai_ReLU[1], moai_z[1]] ∈ MathOptInterface.SOS1{Float64}([1.0, 2.0])
  ├ moai_ReLU[2] ≥ 0
  ├ moai_ReLU[2] ≤ 2
  ├ moai_z[2] ≥ 0
  ├ moai_z[2] ≤ 1
  ├ x[2] - moai_ReLU[2] + moai_z[2] = 0
  └ [moai_ReLU[2], moai_z[2]] ∈ MathOptInterface.SOS1{Float64}([1.0, 2.0])
```
"""
struct ReLUSOS1 <: AbstractPredictor end

output_size(::ReLUQuadratic, input_size) = input_size

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReLUSOS1,
    x::Vector,
)
    cons = Any[]
    y = add_variables(model, x, length(x), "moai_ReLU")
    z = add_variables(model, x, length(x), "moai_z")
    for i in 1:length(x)
        l, u = get_variable_bounds(x[i])
        lb = coalesce(max(0, l), 0)
        set_variable_bounds(cons, y[i], lb, max(0, u); optional = false)
        set_variable_bounds(cons, z[i], 0, max(0, -l); optional = false)
        push!(cons, JuMP.@constraint(model, x[i] == y[i] - z[i]))
        push!(
            cons,
            JuMP.@constraint(model, [y[i], z[i]] in MOI.SOS1([1.0, 2.0])),
        )
    end
    return y, Formulation(predictor, Any[y; z], cons)
end
