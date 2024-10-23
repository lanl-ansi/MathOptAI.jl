# PyTorch

[PyTorch](https://pytorch.org) is a library for machine learning in Python.

## Suupported layers

MathOptAI supports embedding PyTorch models in JuMP if they are an
`nn.Sequential` composed of:

 * `nn.Linear`
 * `nn.ReLU`
 * `nn.Sequential`
 * `nn.Sigmoid`
 * `nn.Tanh`

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
