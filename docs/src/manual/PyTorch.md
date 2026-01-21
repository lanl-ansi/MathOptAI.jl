# PyTorch

[PyTorch](https://pytorch.org) is a library for machine learning in Python.

The upstream documentation is available at
[https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/).

!!! info
    To use PyTorch from MathOptAI, you must first follow the
    [Python integration](@ref) instructions.

## Supported layers

MathOptAI supports embedding a PyTorch models into JuMP if it is a
[`nn.Sequential`](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)
composed of:

 * [`nn.AvgPool2d`](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html)
 * [`nn.Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
 * [`nn.Flatten`](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html)
 * [`nn.GELU`](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html)
 * [`nn.LayerNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
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

Use the `gray_box = true` keyword to embed the network as a vector nonlinear
operator:

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
    config = Dict(:ReLU => MathOptAI.ReLUSOS1),
);
y
formulation
```

## Custom layers

If your PyTorch model contains a custom layer, define a new [`AbstractPredictor`](@ref)
and pass a `config` dictionary that maps the Class object to a callback that
builds the new predictor.

The callback must have the signature `(layer::PythonCall.Py; kwargs...)`. Valid
keyword arguments are currently:

 * `input_size`: the input size of they layer
 * `config`: the `config` dictionary, if needed to convert layers inside the
   custom layer
 * `nn`: a reference to `torch.nn`

You must always have `kwargs...` so that future versions of MathOptAI can add
new keywords in a non-breaking way.

```@repl
using JuMP, PythonCall, MathOptAI
dir = mktempdir()
write(
    joinpath(dir, "custom_model.py"),
    """
    import torch
    class Skip(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, x):
            return self.inner(x) + x
    """,
)
filename = joinpath(dir, "custom_model.pt")
PythonCall.@pyexec(
    (dir, filename) =>
        """
        import sys
        sys.path.insert(0, dir)
        import torch
        from custom_model import Skip
        inner = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.ReLU())
        model = Skip(inner)
        torch.save(model, filename)
        """ => Skip,
)
struct CustomPredictor <: MathOptAI.AbstractPredictor
    p::MathOptAI.Pipeline
end
function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
    predictor::CustomPredictor,
    x::Vector;
    kwargs...,
)
    y, formulation = MathOptAI.add_predictor(model, predictor.p, x; kwargs...)
    @assert length(x) == length(y)
    return y .+ x, formulation
end
model = Model();
@variable(model, x[i in 1:3]);
predictor = MathOptAI.PytorchModel(filename)
function skip_callback(layer::PythonCall.Py; input_size, kwargs...)
    return CustomPredictor(MathOptAI.build_predictor(layer.inner))
end
config = Dict(Skip => skip_callback)
y, formulation = MathOptAI.add_predictor(model, predictor, x; config);
y
formulation
```
