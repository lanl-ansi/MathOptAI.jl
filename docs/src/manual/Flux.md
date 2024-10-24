# Flux

[Flux.jl](https://github.com/FluxML/Flux.jl) is a library for machine learning
in Julia.

## Supported layers

MathOptAI supports embedding Flux models in JuMP if they are a `Flux.Chain`
composed of:

 * `Flux.Dense`
 * `Flux.softmax`
 * `Flux.relu`
 * `Flux.sigmoid`
 * `Flux.softplus`
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
