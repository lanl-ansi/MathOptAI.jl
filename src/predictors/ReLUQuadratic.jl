# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    ReLUQuadratic(; relaxation_parameter = nothing) <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents the relationship:
```math
y = \\max\\{0, x\\}
```
by the reformulation:
```math
\\begin{aligned}
x = y - z \\\\
y \\cdot z = 0 \\\\
y, z \\ge 0
\\end{aligned}
```
If `relaxation_parameter` is set to a value `ϵ`, the constraints become:
```math
\\begin{aligned}
x = y - z \\\\
y \\cdot z \\leq \\epsilon \\\\
y, z \\ge 0
\\end{aligned}
```

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, -1 <= x[i in 1:2] <= i);

julia> f = MathOptAI.ReLUQuadratic()
ReLUQuadratic(nothing)

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
2-element Vector{VariableRef}:
 moai_ReLU[1]
 moai_ReLU[2]

julia> formulation
ReLUQuadratic(nothing)
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
  ├ moai_ReLU[1]*moai_z[1] = 0
  ├ moai_ReLU[2] ≥ 0
  ├ moai_ReLU[2] ≤ 2
  ├ moai_z[2] ≥ 0
  ├ moai_z[2] ≤ 1
  ├ x[2] - moai_ReLU[2] + moai_z[2] = 0
  └ moai_ReLU[2]*moai_z[2] = 0
```
"""
struct ReLUQuadratic <: AbstractPredictor
    relaxation_parameter::Union{Nothing,Float64}
    function ReLUQuadratic(;
        relaxation_parameter::Union{Nothing,Float64} = nothing,
    )
        @assert something(relaxation_parameter, 0.0) >= 0.0
        return new(relaxation_parameter)
    end
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReLUQuadratic,
    x::Vector,
)
    m = length(x)
    cons = Any[]
    bounds = get_variable_bounds.(x)
    y = add_variables(model, predictor, x, m, "moai_ReLU")
    z = add_variables(model, predictor, x, m, "moai_z")
    ϵ = predictor.relaxation_parameter
    for i in 1:m
        l, u = get_variable_bounds(x[i])
        lb = coalesce(max(0, l), 0)
        set_variable_bounds(cons, y[i], lb, max(0, u); optional = false)
        set_variable_bounds(cons, z[i], 0, max(0, -l); optional = false)
        push!(cons, JuMP.@constraint(model, x[i] == y[i] - z[i]))
        if ϵ === nothing
            push!(cons, JuMP.@constraint(model, y[i] * z[i] == 0))
        else
            push!(cons, JuMP.@constraint(model, y[i] * z[i] <= ϵ))
        end
    end
    return y, Formulation(predictor, Any[y; z], cons)
end
