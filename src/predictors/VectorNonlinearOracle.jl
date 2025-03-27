# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    VectorNonlinearOracle(x)

A wrapper struct for creating an `Ipopt.VectorNonlinearOracle`.

!!! warning
    To use [`VectorNonlinearOracle`](@ref), your code must load the `Ipopt` an
    `PythonCall` packages:
    ```julia
    import Ipopt
    import PythonCall
    ```

## Example

```jldoctest
julia> using MathOptAI

julia> using Ipopt, PythonCall  #  This line is important!

julia> predictor = VectorNonlinearOracle(PytorchModel("model.pt"));
```
"""
struct VectorNonlinearOracle{P} <: AbstractPredictor
    predictor::P
end
