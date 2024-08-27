# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    Affine(
        A::Matrix{Float64},
        b::Vector{Float64} = zeros(size(A, 1)),
    ) <: AbstractPredictor

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

julia> y = MathOptAI.add_predictor(model, f, x)
1-element Vector{VariableRef}:
 moai_Affine[1]

julia> print(model)
Feasibility
Subject to
 2 x[1] + 3 x[2] - moai_Affine[1] = 0

julia> y = MathOptAI.add_predictor(model, MathOptAI.ReducedSpace(f), x)
1-element Vector{AffExpr}:
 2 x[1] + 3 x[2]
```
"""
struct Affine <: AbstractPredictor
    A::Matrix{Float64}
    b::Vector{Float64}
end

function Affine(A::Matrix{Float64})
    return Affine(A, zeros(size(A, 1)))
end

function Affine(A::Vector{Float64})
    return Affine(reshape(A, 1, length(A)), [0.0])
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
    JuMP.@constraint(model, predictor.A * x .+ predictor.b .== y)
    return y
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReducedSpace{Affine},
    x::Vector,
)
    A, b = predictor.predictor.A, predictor.predictor.b
    return JuMP.@expression(model, A * x .+ b)
end
