# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

"""
    PytorchModel(filename::String)

A wrapper struct for loading a PyTorch model.

The only supported file extension is `.pt`, where the `.pt` file has been
created using `torch.save(model, filename)`.

## Example

```jldoctest
julia> using PythonCall, MathOptAI

julia> ml_model = PytorchModel("model.pt");
```
"""
struct PytorchModel
    filename::String
end
