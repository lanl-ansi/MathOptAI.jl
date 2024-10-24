# PyTorch

[PyTorch](https://pytorch.org) is a library for machine learning in Python.

The upstream documentation is available at
[https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/).

## Suupported layers

MathOptAI supports embedding a PyTorch models into JuMP if it is a
[`nn.Sequential`](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)
composed of:

 * [`nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
 * [`nn.ReLU`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)
 * [`nn.Sigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)
 * [`nn.Softplus`](https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html)
 * [`nn.Tanh`](https://pytorch.org/docs/stable/generated/torch.nn.Tanhh.html)

## Basic example

Here is an example:

```julia
julia> using JuMP, MathOptAI

julia> model = Model()

julia> @variable(model, x[1:2]);

julia> predictor = MathOptAI.PytorchModel("saved_pytorch_model.pt");

julia> y, _ = MathOptAI.add_predictor(model, predictor, x);
```

## Reduced-space

Use the `reduced_space = true` keyword to formulate a reduced-space model:

```julia
julia> using JuMP, MathOptAI

julia> model = Model()

julia> @variable(model, x[1:2]);

julia> predictor = MathOptAI.PytorchModel("saved_pytorch_model.pt");

julia> y, formulation =
           MathOptAI.add_predictor(model, predictor, x; reduced_space = true);
```

## Gray-box

Use the `gray_box = true` keyword to embed the network as a nonlinear operator:

```julia
julia> using JuMP, MathOptAI

julia> model = Model()

julia> @variable(model, x[1:2]);

julia> predictor = MathOptAI.PytorchModel("saved_pytorch_model.pt");

julia> y, formulation =
           MathOptAI.add_predictor(model, predictor, x; gray_box = true);
```

## Change how layers are formulated

Pass a dictionary to the `config` keyword that maps the `Symbol` name of each
PyTorch layer to a MathOptAI predictor:

```julia
julia> using JuMP, MathOptAI

julia> model = Model()

julia> @variable(model, x[1:2]);

julia> predictor = MathOptAI.PytorchModel("saved_pytorch_model.pt");

julia> y, formulation = MathOptAI.add_predictor(
           model,
           predictor,
           x;
           config = Dict(:ReLU => MathOptAI.ReLUSOS1()),
       );
```
