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
└ constraints [10]
  ├ moai_ReLU[1] ≥ 0
  ├ moai_ReLU[1] ≤ 1
  ├ moai_ReLU[2] ≥ 0
  ├ moai_ReLU[2] ≤ 2
  ├ moai_z[1] ≤ 1
  ├ moai_z[2] ≤ 1
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
    cons = _set_direct_bounds(x -> max(0, x), 0, nothing, x, y)
    z = JuMP.@variable(model, [1:m], lower_bound = 0, base_name = "moai_z")
    _set_bounds_if_finite.(Ref(cons), z, nothing, -first.(bounds))
    append!(cons, JuMP.@constraint(model, x .== y - z))
    formulation = Formulation(predictor, Any[y; z], cons)
    for i in 1:m
        c = JuMP.@constraint(model, [y[i], z[i]] in MOI.SOS1([1.0, 2.0]))
        push!(formulation.constraints, c)
    end
    return y, formulation
end
