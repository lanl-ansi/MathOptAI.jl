# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    ReLUEpigraph() <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents the relationship:
```math
y = \\max\\{0, x\\}
```
by the reformulation:
```math
\\begin{aligned}
y \\ge x \\\\
y \\ge 0
\\end{aligned}
```

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, -1 <= x[i in 1:2] <= i);

julia> f = MathOptAI.ReLUEpigraph()
ReLUEpigraph()

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
2-element Vector{VariableRef}:
 moai_ReLUEpigraph[1]
 moai_ReLUEpigraph[2]

julia> formulation
ReLUEpigraph()
├ variables [2]
│ ├ moai_ReLUEpigraph[1]
│ └ moai_ReLUEpigraph[2]
└ constraints [6]
  ├ -x[1] + moai_ReLUEpigraph[1] ≥ 0
  ├ moai_ReLUEpigraph[1] ≥ 0
  ├ moai_ReLUEpigraph[1] ≤ 1
  ├ -x[2] + moai_ReLUEpigraph[2] ≥ 0
  ├ moai_ReLUEpigraph[2] ≥ 0
  └ moai_ReLUEpigraph[2] ≤ 1
```
"""
struct ReLUEpigraph <: AbstractPredictor end

output_size(::ReLUEpigraph, input_size) = input_size

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReLUEpigraph,
    x::Vector,
)
    y = add_variables(model, x, length(x), "moai_ReLUEpigraph")
    cons = Any[]
    for i in 1:length(x)
        l, u = max.(0, get_variable_bounds(x[i]))
        push!(cons, JuMP.@constraint(model, y[i] >= x[i]))
        set_variable_bounds(cons, y[i], coalesce(l, 0), u; optional = false)
    end
    return y, Formulation(predictor, y, cons)
end
