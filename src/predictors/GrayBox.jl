# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
     GrayBox(
        predictor::P;
        device::String = "cpu",
        hessian::Bool = true,
    ) where {P}

An [`AbstractPredictor`](@ref) that represents the relationship:
```math
y = f(x)
```
as a vector nonlinear operator.

This predictor should not be used directly; it is intended to be used by
extensions like Flux and PyTorch.

## Example

```jldoctest
julia> using JuMP, MathOptAI, Flux

julia> chain = Flux.Chain(Flux.Dense(1 => 16, Flux.relu), Flux.Dense(16 => 1));

julia> model = Model();

julia> @variable(model, x[1:1]);

julia> y, _ = MathOptAI.add_predictor(model, chain, x; gray_box = true);

julia> y
1-element Vector{VariableRef}:
 moai_Flux[1]

julia> print(model)
Feasibility
Subject to
 [x[1], moai_Flux[1]] ∈ VectorNonlinearOracle{Float64}(;
     dimension = 2,
     l = [0.0],
     u = [0.0],
     ...,
 )
```
"""
struct GrayBox{P} <: AbstractPredictor
    predictor::P
    device::String
    hessian::Bool

    function GrayBox(
        predictor::P;
        device::String = "cpu",
        hessian::Bool = true,
    ) where {P}
        return new{P}(predictor, device, hessian)
    end
end

function add_predictor(
    ::JuMP.AbstractModel,
    ::ReducedSpace{<:GrayBox},
    ::Vector,
)
    return error("cannot construct reduced-space formulation of GrayBox")
end
