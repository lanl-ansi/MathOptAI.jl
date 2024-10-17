# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    GrayBox(
        output_size::Function,
        callback::Function;
        has_hessian::Bool = false,
    ) <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents the function ``f(x)`` as a
user-defined nonlinear operator.

## Arguments

 * `output_size(x::Vector):Int`: given an input vector `x`, return the dimension
   of the output vector
 * `callback(x::Vector)::NamedTuple -> (;value, jacobian[, hessian])`: given an
   input vector `x`, return a `NamedTuple` that computes the primal value and
   Jacobian of the output value with respect to the input. `jacobian[j, i]` is
   the partial derivative of `value[j]` with respect to `x[i]`.
 * `has_hessian`: if `true`, the `callback` additionally contains a field
   `hessian`, which is an `N × N × M` matrix, where `hessian[i, j, k]` is the
   partial derivative of `value[k]` with respect to `x[i]` and `x[j]`.

## Example

```jldoctest; filter=r"##[0-9]+"
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = MathOptAI.GrayBox(
           x -> 2,
           x -> (value = x.^2, jacobian = [2 * x[1] 0.0; 0.0 2 * x[2]]),
       );

julia> y, formulation = MathOptAI.add_predictor(model, f, x);

julia> y
2-element Vector{VariableRef}:
 moai_GrayBox[1]
 moai_GrayBox[2]

julia> formulation
GrayBox
├ variables [2]
│ ├ moai_GrayBox[1]
│ └ moai_GrayBox[2]
└ constraints [2]
  ├ op_##330(x[1], x[2]) - moai_GrayBox[1] = 0
  └ op_##331(x[1], x[2]) - moai_GrayBox[2] = 0

julia> y, formulation = MathOptAI.add_predictor(model, MathOptAI.ReducedSpace(f), x);

julia> y
2-element Vector{NonlinearExpr}:
 op_##332(x[1], x[2])
 op_##333(x[1], x[2])

julia> formulation
ReducedSpace(GrayBox)
├ variables [0]
└ constraints [0]
```
"""
struct GrayBox{F<:Function,G<:Function} <: AbstractPredictor
    output_size::F
    callback::G
    has_hessian::Bool

    function GrayBox(
        output_size::F,
        callback::G;
        has_hessian::Bool = false,
    ) where {F<:Function,G<:Function}
        return new{F,G}(output_size, callback, has_hessian)
    end
end

Base.show(io::IO, ::GrayBox) = print(io, "GrayBox")

function add_predictor(model::JuMP.AbstractModel, predictor::GrayBox, x::Vector)
    op, _ = add_predictor(model, ReducedSpace(predictor), x)
    y = JuMP.@variable(model, [1:length(op)], base_name = "moai_GrayBox")
    cons = JuMP.@constraint(model, op .== y)
    return y, Formulation(predictor, y, cons)
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReducedSpace{<:GrayBox},
    x::Vector,
)
    last_x, cache = nothing, nothing
    function update(x)
        if x != last_x
            cache = predictor.predictor.callback(x)
            last_x = x
        end
        return
    end
    function f(i::Int, x...)::Float64
        update(x)
        return cache.value[i]
    end
    function ∇f(g::AbstractVector{Float64}, i::Int, x...)
        update(x)
        g .= cache.jacobian[i, :]
        return
    end
    function ∇²f(H::AbstractMatrix{Float64}, k::Int, x...)
        update(x)
        for j in 1:length(x), i in j:length(x)
            H[i, j] = cache.hessian[i, j, k]
        end
        return
    end
    y = map(1:predictor.predictor.output_size(x)) do i
        callbacks = if predictor.predictor.has_hessian
            ∇²fi = (H, x...) -> ∇²f(H, i, x...)
            ((x...) -> f(i, x...), (g, x...) -> ∇f(g, i, x...), ∇²fi)
        else
            ((x...) -> f(i, x...), (g, x...) -> ∇f(g, i, x...))
        end
        name = Symbol("op_$(gensym())")
        op_i = JuMP.add_nonlinear_operator(model, length(x), callbacks...; name)
        return op_i(x...)
    end
    return y, Formulation(predictor)
end
