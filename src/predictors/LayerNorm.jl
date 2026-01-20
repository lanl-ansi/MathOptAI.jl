# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    LayerNorm(
        shape::NTuple{N,Int};
        input_size::Tuple{Int,Int,Int},
        eps::Float64 = 1e-5,
    ) where {N}

An [`AbstractPredictor`](@ref) that represents the relationship:
```math
y = \\frac{x - E[x]}{\\sqrt{Var(x) + \\eps}}
```
where `E` and `Var` are computed over the first `shape` dimensions of `x`.

!!! Note
    This layer does **not** implement the affine scaling seen in some layers.
    Apply [`Scale`](@ref) to the outputs of this predictor instead.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2, 1:3]);

julia> f = MathOptAI.LayerNorm(
           (2,);
           eps = 0.0,
           input_size = (2, 3, 1),
           weight = [1.0, 2.0],
           bias = [0.5, 0.6],
       )
LayerNorm{Float64, 1}((2, 3, 1), (2,), 0.0, [1.0, 2.0], [0.5, 0.6])

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
6-element Vector{VariableRef}:
 moai_LayerNorm[1]
 moai_LayerNorm[2]
 moai_LayerNorm[3]
 moai_LayerNorm[4]
 moai_LayerNorm[5]
 moai_LayerNorm[6]

julia> formulation
LayerNorm{Float64, 1}((2, 3, 1), (2,), 0.0, [1.0, 2.0], [0.5, 0.6])
├ variables [3]
│ ├ VariableRef[moai_LayerNorm_μ[1], moai_LayerNorm_μ[2], moai_LayerNorm_μ[3]]
│ ├ VariableRef[moai_LayerNorm_σ[1], moai_LayerNorm_σ[2], moai_LayerNorm_σ[3]]
│ └ VariableRef[moai_LayerNorm[1], moai_LayerNorm[2], moai_LayerNorm[3], moai_LayerNorm[4], moai_LayerNorm[5], moai_LayerNorm[6]]
└ constraints [15]
  ├ moai_LayerNorm_σ[1] ≥ 0
  ├ moai_LayerNorm_σ[2] ≥ 0
  ├ moai_LayerNorm_σ[3] ≥ 0
  ├ -x[1,1] - x[2,1] + 2 moai_LayerNorm_μ[1] = 0
  ├ -0.5 x[1,1]² + moai_LayerNorm_μ[1]*x[1,1] - 0.5 x[2,1]² + moai_LayerNorm_μ[1]*x[2,1] - moai_LayerNorm_μ[1]² + moai_LayerNorm_σ[1]² = 0
  ├ moai_LayerNorm_σ[1]*moai_LayerNorm[1] - x[1,1] + moai_LayerNorm_μ[1] - 0.5 moai_LayerNorm_σ[1] = 0
  ├ moai_LayerNorm_σ[1]*moai_LayerNorm[2] - 2 x[2,1] + 2 moai_LayerNorm_μ[1] - 0.6 moai_LayerNorm_σ[1] = 0
  ├ -x[1,2] - x[2,2] + 2 moai_LayerNorm_μ[2] = 0
  ├ -0.5 x[1,2]² + moai_LayerNorm_μ[2]*x[1,2] - 0.5 x[2,2]² + moai_LayerNorm_μ[2]*x[2,2] - moai_LayerNorm_μ[2]² + moai_LayerNorm_σ[2]² = 0
  ├ moai_LayerNorm_σ[2]*moai_LayerNorm[3] - x[1,2] + moai_LayerNorm_μ[2] - 0.5 moai_LayerNorm_σ[2] = 0
  ├ moai_LayerNorm_σ[2]*moai_LayerNorm[4] - 2 x[2,2] + 2 moai_LayerNorm_μ[2] - 0.6 moai_LayerNorm_σ[2] = 0
  ├ -x[1,3] - x[2,3] + 2 moai_LayerNorm_μ[3] = 0
  ├ -0.5 x[1,3]² + moai_LayerNorm_μ[3]*x[1,3] - 0.5 x[2,3]² + moai_LayerNorm_μ[3]*x[2,3] - moai_LayerNorm_μ[3]² + moai_LayerNorm_σ[3]² = 0
  ├ moai_LayerNorm_σ[3]*moai_LayerNorm[5] - x[1,3] + moai_LayerNorm_μ[3] - 0.5 moai_LayerNorm_σ[3] = 0
  └ moai_LayerNorm_σ[3]*moai_LayerNorm[6] - 2 x[2,3] + 2 moai_LayerNorm_μ[3] - 0.6 moai_LayerNorm_σ[3] = 0
```
"""
struct LayerNorm{T,N} <: AbstractPredictor
    input_size::Tuple{Int,Int,Int}
    shape::NTuple{N,Int}
    eps::T
    weight::Array{T,N}
    bias::Array{T,N}

    function LayerNorm(
        shape::NTuple{N,Int};
        input_size::Tuple{Int,Int,Int},
        eps::T = 1e-5,
        weight::Array{T,N} = ones(T, shape),
        bias::Array{T,N} = zeros(T, shape),
    ) where {N,T}
        @assert size(weight) == size(bias) == shape
        return new{T,N}(input_size, shape, eps, weight, bias)
    end
end

output_size(::LayerNorm, input_size) = input_size

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::LayerNorm{T,N},
    x::Vector,
) where {T,N}
    ε, γ, β = predictor.eps, predictor.weight, predictor.bias
    n = prod(predictor.input_size[1:N])
    m = prod(predictor.input_size[(N+1):end])
    y = add_variables(model, x, length(x), "moai_LayerNorm")
    μ = add_variables(model, x, m, "moai_LayerNorm_μ")
    σ = add_variables(model, x, m, "moai_LayerNorm_σ")
    cons = Any[]
    for i in 1:m
        set_variable_bounds(cons, σ[i], ε, missing; optional = false)
    end
    # It's important to start this with a non-zero start value, otherwise the
    # starting point has a /0 or equivalently, (0 * y).
    set_variable_start.(σ, max(ε, 1.0))
    for k in 1:m
        offset = n * (k - 1)
        # μ = E[X]
        c_μ = JuMP.@constraint(model, n * μ[k] == sum(x[offset+i] for i in 1:n))
        push!(cons, c_μ)
        # σ = √(Var[X]+ε) <--> σ² = Var[X] + ε
        c_σ = JuMP.@constraint(
            model,
            σ[k]^2 == sum((x[offset+i] - μ[k])^2 for i in 1:n) / n + ε,
        )
        push!(cons, c_σ)
        #       (x - μ)
        # y = ----------- * γ + β
        #     √(Var[X]+ε)
        for i in 1:n
            yi_con = JuMP.@constraint(
                model,
                σ[k] * (y[offset+i] - β[i]) == (x[offset+i] - μ[k]) * γ[i],
            )
            push!(cons, yi_con)
        end
    end
    return y, Formulation(predictor, [μ, σ, y], cons)
end
