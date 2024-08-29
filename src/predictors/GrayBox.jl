# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    GrayBox(
        output_size::Function,
        with_jacobian::Function,
    ) <: AbstractPredictor

An [`AbstractPredictor`](@ref) that represents the function ``f(x)`` as a
user-defined nonlinear operator.

## arguments

 * `output_size(x::Vector):Int`: given an input vector `x`, return the dimension
   of the output vector
 * `with_jacobian(x::Vector)::NamedTuple -> (;value, jacobian)`: given an input
   vector `x`, return a `NamedTuple` that computes the primal value and Jacobian
   of the output value with respect to the input. `jacobian[j, i]` is the
   partial derivative of `value[j]` with respect to `x[i]`.

## Example

```jldoctest; filter=r"##[0-9]+"
julia> using JuMP, MathOptAI

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = MathOptAI.GrayBox(
           x -> 2,
           x -> (value = x.^2, jacobian = [2 * x[1] 0.0; 0.0 2 * x[2]]),
       );

julia> y = MathOptAI.add_predictor(model, f, x)
1-element Vector{VariableRef}:
 moai_GrayBox[1]
 moai_GrayBox[2]

julia> print(model)
Feasibility
Subject to
 op_##238(x[1], x[2]) - moai_GrayBox[1] = 0
 op_##239(x[1], x[2]) - moai_GrayBox[2] = 0

julia> y = MathOptAI.add_predictor(model, MathOptAI.ReducedSpace(f), x)
2-element Vector{NonlinearExpr}:
 op_##240(x[1], x[2])
 op_##241(x[1], x[2])
```
"""
struct GrayBox{F<:Function,G<:Function} <: AbstractPredictor
    output_size::F
    with_jacobian::G
end

function add_predictor(model::JuMP.AbstractModel, predictor::GrayBox, x::Vector)
    op = add_predictor(model, ReducedSpace(predictor), x)
    y = JuMP.@variable(model, [1:length(op)], base_name = "moai_GrayBox")
    JuMP.@constraint(model, op .== y)
    return y
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReducedSpace{<:GrayBox},
    x::Vector,
)
    last_x, cache = nothing, nothing
    function update(x)
        if x != last_x
            cache = predictor.predictor.with_jacobian(x)
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
    return map(1:predictor.predictor.output_size(x)) do i
        op_i = JuMP.add_nonlinear_operator(
            model,
            length(x),
            (x...) -> f(i, x...),
            (g, x...) -> ∇f(g, i, x...);
            name = Symbol("op_$(gensym())"),
        )
        return op_i(x...)
    end
end
