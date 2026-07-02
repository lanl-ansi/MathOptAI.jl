# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    SoftPlusConicEpigraph(; beta::Float64 = 1.0) <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents the relationship:
```math
y = \\frac{1}{\\beta} \\log(1 + e^{\\beta x})
```
by the reformulation:
```math
\\begin{aligned}
u_1 + u_2 & \\le 1 \\\\
(u_1, 1, \\beta(x - y)) & \\in \\mathcal{K}_{\\exp} \\\\
(u_2, 1, - \\beta y) & \\in \\mathcal{K}_{\\exp}
\\end{aligned}
```

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, -1 <= x[i in 1:2] <= i);

julia> f = MathOptAI.SoftPlusConicEpigraph()
SoftPlusConicEpigraph(1.0)

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
2-element Vector{VariableRef}:
 moai_SoftPlusConicEpigraph_y[1]
 moai_SoftPlusConicEpigraph_y[2]

julia> formulation
SoftPlusConicEpigraph(1.0)
├ variables [6]
│ ├ moai_SoftPlusConicEpigraph_y[1]
│ ├ moai_SoftPlusConicEpigraph_y[2]
│ ├ moai_SoftPlusConicEpigraph_u1[1]
│ ├ moai_SoftPlusConicEpigraph_u1[2]
│ ├ moai_SoftPlusConicEpigraph_u2[1]
│ └ moai_SoftPlusConicEpigraph_u2[2]
└ constraints [6]
  ├ [x[1] - moai_SoftPlusConicEpigraph_y[1], 1, moai_SoftPlusConicEpigraph_u1[1]] ∈ MathOptInterface.ExponentialCone()
  ├ [-moai_SoftPlusConicEpigraph_y[1], 1, moai_SoftPlusConicEpigraph_u2[1]] ∈ MathOptInterface.ExponentialCone()
  ├ moai_SoftPlusConicEpigraph_u1[1] + moai_SoftPlusConicEpigraph_u2[1] ≤ 1
  ├ [x[2] - moai_SoftPlusConicEpigraph_y[2], 1, moai_SoftPlusConicEpigraph_u1[2]] ∈ MathOptInterface.ExponentialCone()
  ├ [-moai_SoftPlusConicEpigraph_y[2], 1, moai_SoftPlusConicEpigraph_u2[2]] ∈ MathOptInterface.ExponentialCone()
  └ moai_SoftPlusConicEpigraph_u1[2] + moai_SoftPlusConicEpigraph_u2[2] ≤ 1
```
"""
struct SoftPlusConicEpigraph <: MathOptAI.AbstractPredictor
    beta::Float64
    SoftPlusConicEpigraph(; beta::Float64 = 1.0) = new(beta)
end

output_size(::SoftPlusConicEpigraph, input_size) = input_size

function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
    predictor::SoftPlusConicEpigraph,
    x::Vector,
)
    cons = Any[]
    u1 = add_variables(model, x, length(x), "moai_SoftPlusConicEpigraph_u1")
    u2 = add_variables(model, x, length(x), "moai_SoftPlusConicEpigraph_u2")
    y = add_variables(model, x, length(x), "moai_SoftPlusConicEpigraph_y")
    for i in 1:length(x)
        push!(
            cons,
            JuMP.@constraint(
                model,
                [predictor.beta * (x[i] - y[i]), 1, u1[i]] in
                MOI.ExponentialCone()
            )
        )
        push!(
            cons,
            JuMP.@constraint(
                model,
                [-predictor.beta * y[i], 1, u2[i]] in MOI.ExponentialCone()
            )
        )
        push!(cons, JuMP.@constraint(model, u1[i] + u2[i] <= 1))
    end
    return y, Formulation(predictor, Any[y; u1; u2], cons)
end
