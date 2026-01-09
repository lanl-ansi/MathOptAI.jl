# PyTorch

[PyTorch](https://pytorch.org) is a library for machine learning in Python.

The upstream documentation is available at
[https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/).

## Supported layers

MathOptAI supports embedding a PyTorch models into JuMP if it is a
[`nn.Sequential`](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)
composed of:

 * [`nn.AvgPool2d`](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html)
 * [`nn.Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
 * [`nn.Flatten`](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html)
 * [`nn.GELU`](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html)
 * [`nn.LeakyReLU`](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html)
 * [`nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
 * [`nn.MaxPool2d`](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
 * [`nn.ReLU`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)
 * [`nn.Sigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)
 * [`nn.Softmax`](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html)
 * [`nn.Softplus`](https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html)
 * [`nn.Tanh`](https://pytorch.org/docs/stable/generated/torch.nn.Tanhh.html)

## File format

Use [`torch.save`](https://pytorch.org/docs/stable/generated/torch.save.html) to
save a trained PyTorch model to a `.pt` file:

```python
#!/usr/bin/python3
import torch
model = torch.nn.Sequential(
    torch.nn.Linear(1, 2),
    torch.nn.ReLU(),
    torch.nn.Linear(2, 1),
)
torch.save(model, "saved_pytorch_model.pt")
```

## Python integration

MathOptAI uses [PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl) to call
from Julia into Python.

To use [`PytorchModel`](@ref) your code must load the `PythonCall` package:
```julia
import PythonCall
```

PythonCall uses [CondaPkg.jl](https://github.com/JuliaPy/CondaPkg.jl) to manage
Python dependencies. See [CondaPkg.jl](https://github.com/JuliaPy/CondaPkg.jl)
for more control over how to link Julia to an existing Python environment. For
example, if you have an existing Python installation (with PyTorch installed),
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

## Basic example

Use [`MathOptAI.add_predictor`](@ref) to embed a PyTorch model into a JuMP
model:

```@repl
using JuMP, MathOptAI, PythonCall
model = Model();
@variable(model, x[1:1]);
predictor = MathOptAI.PytorchModel("saved_pytorch_model.pt");
y, formulation = MathOptAI.add_predictor(model, predictor, x);
y
formulation
```

## Reduced-space

Use the `reduced_space = true` keyword to formulate a reduced-space model:

```@repl
using JuMP, MathOptAI, PythonCall
model = Model();
@variable(model, x[1:1]);
predictor = MathOptAI.PytorchModel("saved_pytorch_model.pt");
y, formulation =
    MathOptAI.add_predictor(model, predictor, x; reduced_space = true);
y
formulation
```

## Gray-box

Use the `gray_box = true` keyword to embed the network as a nonlinear operator:

```@repl
using JuMP, MathOptAI, PythonCall
model = Model();
@variable(model, x[1:1]);
predictor = MathOptAI.PytorchModel("saved_pytorch_model.pt");
y, formulation =
    MathOptAI.add_predictor(model, predictor, x; gray_box = true);
y
formulation
```

## [VectorNonlinearOracle](@id pytorch-vector-nonlinear-oracle)

Use the `vector_nonlinear_oracle = true` keyword to embed the network as a
vector nonlinear operator:

```@repl
using JuMP, MathOptAI, PythonCall
model = Model();
@variable(model, x[1:1]);
predictor = MathOptAI.PytorchModel("saved_pytorch_model.pt");
y, formulation = MathOptAI.add_predictor(
    model,
    predictor,
    x;
    vector_nonlinear_oracle = true,
);
y
formulation
```

## Change how layers are formulated

Pass a dictionary to the `config` keyword that maps the `Symbol` name of each
PyTorch layer to a MathOptAI predictor:

```@repl
using JuMP, MathOptAI, PythonCall
model = Model();
@variable(model, x[1:1]);
predictor = MathOptAI.PytorchModel("saved_pytorch_model.pt");
y, formulation = MathOptAI.add_predictor(
    model,
    predictor,
    x;
    config = Dict(:ReLU => MathOptAI.ReLUSOS1()),
);
y
formulation
```
