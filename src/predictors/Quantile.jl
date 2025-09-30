# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    Quantile{D}(distribution::D, quantiles::Vector{Float64}) where {D}

An [`AbstractPredictor`](@ref) that represents the `quantiles` of `distribution`.

## Example

```jldoctest
julia> using JuMP, Distributions, MathOptAI

julia> model = Model();

julia> @variable(model, 1 <= x <= 2);

julia> predictor = MathOptAI.Quantile([0.1, 0.9]) do x
           return Distributions.Normal(x, 3 - x)
       end
Quantile(_, [0.1, 0.9])

julia> y, formulation = MathOptAI.add_predictor(model, predictor, [x]);

julia> y
2-element Vector{VariableRef}:
 moai_quantile[1]
 moai_quantile[2]

julia> formulation
Quantile(_, [0.1, 0.9])
├ variables [2]
│ ├ moai_quantile[1]
│ └ moai_quantile[2]
└ constraints [2]
  ├ moai_quantile[1] - op_quantile_0.1(x) = 0
  └ moai_quantile[2] - op_quantile_0.9(x) = 0
```
"""
struct Quantile{D} <: AbstractPredictor
    distribution::D
    quantiles::Vector{Float64}
end

function Base.show(io::IO, q::Quantile)
    return print(io, "Quantile(_, $(q.quantiles))")
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::Quantile,
    x::Vector,
)
    M, N = length(x), length(predictor.quantiles)
    y = add_variables(model, x, N, "moai_quantile")
    quantile(q, x...) = Distributions.quantile(predictor.distribution(x...), q)
    cons = Any[]
    for (qi, yi) in zip(predictor.quantiles, y)
        op_i = JuMP.add_nonlinear_operator(
            model,
            M,
            (x...) -> quantile(qi, x...);
            name = Symbol("op_quantile_$qi"),
        )
        push!(cons, JuMP.@constraint(model, yi == op_i(x...)))
    end
    return y, Formulation(predictor, y, cons)
end
