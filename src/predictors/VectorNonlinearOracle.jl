# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    VectorNonlinearOracle(x)

A wrapper struct for creating an `Ipopt.VectorNonlinearOracle`.

!!! warning
    To use [`VectorNonlinearOracle`](@ref), your code must load the `Ipopt`
    package.
    ```julia
    import Ipopt
    ```
"""
struct VectorNonlinearOracle{P} <: AbstractPredictor
    predictor::P
    device::String
    hessian::Bool

    function VectorNonlinearOracle(
        predictor::P;
        device::String = "cpu",
        hessian::Bool = true,
    ) where {P}
        return new{P}(predictor, device, hessian)
    end
end

function add_predictor(
    model::JuMP.AbstractModel,
    predictor::ReducedSpace{<:VectorNonlinearOracle},
    x::Vector;
)
    error("cannot construct reduced-space formulation of VectorNonlinearOracle")
    return
end
