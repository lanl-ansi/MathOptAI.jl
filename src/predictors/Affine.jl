# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    Affine(
        A::Matrix{T},
        b::Vector{T} = zeros(T, size(A, 1)),
    ) where {T} <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents the relationship:
```math
y = A x + b
```

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, 0 <= x[i in 1:2] <= i);

julia> f = MathOptAI.Affine([2.0 3.0], [4.0])
Affine(A, b) [input: 2, output: 1]

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
1-element Vector{VariableRef}:
 moai_Affine[1]

julia> formulation
Affine(A, b) [input: 2, output: 1]
├ variables [1]
│ └ moai_Affine[1]
└ constraints [3]
  ├ moai_Affine[1] ≥ 4
  ├ moai_Affine[1] ≤ 12
  └ 2 x[1] + 3 x[2] - moai_Affine[1] = -4

julia> y, formulation =
           MathOptAI.add_predictor(model, MathOptAI.ReducedSpace(f), x);

julia> y
1-element Vector{AffExpr}:
 2 x[1] + 3 x[2] + 4

julia> formulation
ReducedSpace(Affine(A, b) [input: 2, output: 1])
├ variables [0]
└ constraints [0]
```
"""
struct Affine{T} <: AbstractPredictor
    A::Matrix{T}
    b::Vector{T}

    function Affine{T}(A::Matrix{T}, b::Vector{T}) where {T}
        if size(A, 1) != length(b)
            msg = "[Affine] the `A` matrix must have the same number of rows as the `b` vector. Got `$(size(A, 1))` and `$(length(b))`."
            throw(DimensionMismatch(msg))
        end
        return new(A, b)
    end
end

function Affine(A::AbstractMatrix{U}, b::AbstractVector{V}) where {U,V}
    T = promote_type(U, V)
    return Affine{T}(convert(Matrix{T}, A), convert(Vector{T}, b))
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

function _check_dimension(predictor::Affine, x::Vector)
    m, n = size(predictor.A, 2), length(x)
    if m != n
        msg = "[Affine] mismatch between the length of the input `x` ($n) and the number of columns of the `Affine` predictor ($m)."
        throw(DimensionMismatch(msg))
    end
    return
end

function (predictor::Affine)(x::Vector)
    _check_dimension(predictor, x)
    return predictor.A * x .+ predictor.b
end

function add_predictor(model::JuMP.AbstractModel, predictor::Affine, x::Vector)
    _check_dimension(predictor, x)
    m = size(predictor.A, 1)
    y = add_variables(model, x, m, "moai_Affine")
    bounds = get_variable_bounds.(x)
    cons = Any[]
    for i in 1:size(predictor.A, 1)
        y_lb, y_ub = predictor.b[i], predictor.b[i]
        for j in 1:size(predictor.A, 2)
            a_ij = predictor.A[i, j]
            lb, ub = bounds[j]
            y_ub += a_ij * ifelse(a_ij >= 0, ub, lb)
            y_lb += a_ij * ifelse(a_ij >= 0, lb, ub)
        end
        set_variable_bounds(cons, y[i], y_lb, y_ub; optional = true)
    end
    append!(cons, JuMP.@constraint(model, predictor(x) .== y))
    return y, Formulation(predictor, y, cons)
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReducedSpace{<:Affine},
    x::Vector,
)
    return predictor.predictor(x), Formulation(predictor)
end
