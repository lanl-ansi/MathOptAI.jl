# Flux.jl

[Flux.jl](https://github.com/FluxML/Flux.jl) is a library for machine learning
in Julia.

The upstream documentation is available at
[https://fluxml.ai/Flux.jl/stable/](https://fluxml.ai/Flux.jl/stable/).

## Supported layers

MathOptAI supports embedding a Flux model into JuMP if it is a
[`Flux.Chain`](https://fluxml.ai/Flux.jl/stable/reference/models/layers/#Flux.Chain)
composed of:

  * [`Flux.Dense`](https://fluxml.ai/Flux.jl/stable/reference/models/layers/#Flux.Dense)
  * [`Flux.Scale`](https://fluxml.ai/Flux.jl/stable/reference/models/layers/#Flux.Scale)
  * [`Flux.relu`](https://fluxml.ai/Flux.jl/stable/reference/models/activation/#NNlib.relu)
  * [`Flux.sigmoid`](https://fluxml.ai/Flux.jl/stable/reference/models/activation/#NNlib.sigmoid)
  * [`Flux.softmax`](https://fluxml.ai/Flux.jl/stable/reference/models/nnlib/#NNlib.softmax)
  * [`Flux.softplus`](https://fluxml.ai/Flux.jl/stable/reference/models/activation/#NNlib.softplus)
  * `Flux.tanh`

## Basic example

Use [`MathOptAI.add_predictor`](@ref) to embed a `Flux.Chain` into a JuMP model:

```@repl
using JuMP, Flux, MathOptAI
predictor = Flux.Chain(Flux.Dense(1 => 2, Flux.relu), Flux.Dense(2 => 1));
model = Model();
@variable(model, x[1:1]);
y, formulation = MathOptAI.add_predictor(model, predictor, x);
y
formulation
```

## Reduced-space

Use the `reduced_space = true` keyword to formulate a reduced-space model:

```@repl
using JuMP, Flux, MathOptAI
predictor = Flux.Chain(Flux.Dense(1 => 2, Flux.relu), Flux.Dense(2 => 1));
model = Model();
@variable(model, x[1:1]);
y, formulation =
    MathOptAI.add_predictor(model, predictor, x; reduced_space = true);
y
formulation
```

## Gray-box

Use the `gray_box = true` keyword to embed the network as a nonlinear operator:

```@repl
using JuMP, Flux, MathOptAI
predictor = Flux.Chain(Flux.Dense(1 => 2, Flux.relu), Flux.Dense(2 => 1));
model = Model();
@variable(model, x[1:1]);
y, formulation =
    MathOptAI.add_predictor(model, predictor, x; gray_box = true);
y
formulation
```

## VectorNonlinearOracle

Use the `vector_nonlinear_oracle = true` keyword to embed the network as a
vector nonlinear operator:

```@repl
using JuMP, Flux, MathOptAI
predictor = Flux.Chain(Flux.Dense(1 => 2, Flux.relu), Flux.Dense(2 => 1));
model = Model();
@variable(model, x[1:1]);
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

Pass a dictionary to the `config` keyword that maps Flux activation functions to
a MathOptAI predictor:

```@repl
using JuMP, Flux, MathOptAI
predictor = Flux.Chain(Flux.Dense(1 => 2, Flux.relu), Flux.Dense(2 => 1));
model = Model();
@variable(model, x[1:1]);
y, formulation = MathOptAI.add_predictor(
    model,
    predictor,
    x;
    config = Dict(Flux.relu => MathOptAI.ReLUSOS1()),
);
y
formulation
```
