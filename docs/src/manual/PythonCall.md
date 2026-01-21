# Python integration

MathOptAI uses [PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl) to call
from Julia into Python.

To use [`PytorchModel`](@ref) your code must load the `PythonCall` package:
```julia
import PythonCall
```

PythonCall uses [CondaPkg.jl](https://github.com/JuliaPy/CondaPkg.jl) to manage
Python dependencies. See [CondaPkg.jl](https://github.com/JuliaPy/CondaPkg.jl)
for more control over how to link Julia to an existing Python environment.

If you have an existing Python installation (with PyTorch installed),
and it is available in the current Conda environment, do:

```julia
ENV["JULIA_CONDAPKG_BACKEND"] = "Current"
import PythonCall
```

If the Python installation can be found on the path and it is not in a Conda
environment, do:

```julia
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
import PythonCall
```

If `python` is not on your path, you may additionally need to set
`JULIA_PYTHONCALL_EXE`, for example, do:

```julia
ENV["JULIA_PYTHONCALL_EXE"] = "python3"
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
import PythonCall
```
