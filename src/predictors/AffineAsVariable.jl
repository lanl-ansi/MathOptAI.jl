# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    AffineAsVariable(
        A::Matrix{T},
        b::Vector{T} = zeros(T, size(A, 1)),
    ) where {T} <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents the relationship:
```math
y = A x + b
```
where `A` and `b` are added as decision variables, and their primal start is set
from the input data.

## Example

```jldoctest
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, 0 <= x[i in 1:2] <= i);

julia> f = MathOptAI.AffineAsVariable([2.0 3.0], [4.0])
AffineAsVariable(A, b) [input: 2, output: 1]

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
1-element Vector{VariableRef}:
 moai_AffineAsVariable_y[1]

julia> formulation
AffineAsVariable(A, b) [input: 2, output: 1]
├ variables [4]
│ ├ moai_AffineAsVariable_y[1]
│ ├ moai_AffineAsVariable_A[1]
│ ├ moai_AffineAsVariable_A[2]
│ └ moai_AffineAsVariable_b[1]
└ constraints [1]
  └ moai_AffineAsVariable_A[1]*x[1] + moai_AffineAsVariable_A[2]*x[2] - moai_AffineAsVariable_y[1] + moai_AffineAsVariable_b[1] = 0
```
"""
struct AffineAsVariable{T} <: AbstractPredictor
    A::Matrix{T}
    b::Vector{T}

    function AffineAsVariable(A::Matrix{T}, b::Vector{T}) where {T}
        if size(A, 1) != length(b)
            msg = "[AffineAsVariable] the `A` matrix must have the same number of rows as the `b` vector. Got `$(size(A, 1))` and `$(length(b))`."
            throw(DimensionMismatch(msg))
        end
        return new{T}(A, b)
    end
end

function Base.show(io::IO, p::AffineAsVariable)
    m, n = size(p.A)
    return print(io, "AffineAsVariable(A, b) [input: $n, output: $m]")
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::AffineAsVariable,
    x::Vector,
)
    m, n = size(predictor.A)
    if n != length(x)
        msg = "[AffineAsVariable] mismatch between the length of the input `x` ($(length(x))) and the number of columns of the `AffineAsVariable` predictor ($m)."
        throw(DimensionMismatch(msg))
    end
    y = add_variables(model, x, m, "moai_AffineAsVariable_y")
    A_vec = add_variables(model, x, m * n, "moai_AffineAsVariable_A")
    b = add_variables(model, x, m, "moai_AffineAsVariable_b")
    A = reshape(A_vec, m, n)
    set_variable_start.(A, predictor.A)
    set_variable_start.(b, predictor.b)
    cons = JuMP.@constraint(model, A * x .+ b .== y)
    return y, Formulation(predictor, [y; A_vec; b], cons)
end
