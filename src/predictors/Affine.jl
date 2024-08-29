# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    Affine(
        A::Matrix{T},
        b::Vector{T} = zeros(T, size(A, 1)),
    ) where {T} <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents the affine relationship:
```math
f(x) = A x + b
```

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = MathOptAI.Affine([2.0, 3.0])
Affine(A, b) [input: 2, output: 1]

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
1-element Vector{VariableRef}:
 moai_Affine[1]

julia> formulation
Affine(A, b) [input: 2, output: 1]
├ variables [1]
│ └ moai_Affine[1]
└ constraints [1]
  └ 2 x[1] + 3 x[2] - moai_Affine[1] = 0

julia> y, formulation =
           MathOptAI.add_predictor(model, MathOptAI.ReducedSpace(f), x);

julia> y
1-element Vector{AffExpr}:
 2 x[1] + 3 x[2]

julia> formulation
ReducedSpace(Affine(A, b) [input: 2, output: 1])
├ variables [0]
└ constraints [0]
```
"""
struct Affine{T} <: AbstractPredictor
    A::Matrix{T}
    b::Vector{T}
end

function Affine(A::Matrix{T}) where {T}
    return Affine{T}(A, zeros(T, size(A, 1)))
end

function Affine(A::Vector{T}) where {T}
    return Affine{T}(reshape(A, 1, length(A)), [zero(T)])
end

function Base.show(io::IO, p::Affine)
    m, n = size(p.A)
    return print(io, "Affine(A, b) [input: $n, output: $m]")
end

function add_predictor(model::JuMP.AbstractModel, predictor::Affine, x::Vector)
    m = size(predictor.A, 1)
    y = JuMP.@variable(model, [1:m], base_name = "moai_Affine")
    bounds = _get_variable_bounds.(x)
    for i in 1:size(predictor.A, 1)
        y_lb, y_ub = predictor.b[i], predictor.b[i]
        for j in 1:size(predictor.A, 2)
            a_ij = predictor.A[i, j]
            lb, ub = bounds[j]
            y_ub += a_ij * ifelse(a_ij >= 0, ub, lb)
            y_lb += a_ij * ifelse(a_ij >= 0, lb, ub)
        end
        _set_bounds_if_finite(y[i], y_lb, y_ub)
    end
    cons = JuMP.@constraint(model, predictor.A * x .+ predictor.b .== y)
    return y, SimpleFormulation(predictor, y, cons)
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReducedSpace{<:Affine},
    x::Vector,
)
    A, b = predictor.predictor.A, predictor.predictor.b
    y = JuMP.@expression(model, A * x .+ b)
    return y, SimpleFormulation(predictor)
end
