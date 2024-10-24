# Lux

[Lux.jl](https://github.com/LuxDL/Lux.jl) is a library for machine learning in
Julia.

## Supported layers

MathOptAI supports embedding Lux models in JuMP if they are a chain composed
of:

 * `Lux.Dense`
 * `Lux.relu`
 * `Lux.sigmoid`
 * `Lux.softplus`
 * `Lux.tanh`

## Basic example

Use [`MathOptAI.add_predictor`](@ref) to embed a tuple (containing the
`Lux.Chain`, the `parameters`, and the `state`) into a JuMP model:

```@repl
using JuMP, Lux, MathOptAI, Random
rng = Random.MersenneTwister();
chain = Lux.Chain(Lux.Dense(1 => 2, Lux.relu), Lux.Dense(2 => 1))
parameters, state = Lux.setup(rng, chain);
predictor = (chain, parameters, state);
model = Model();
@variable(model, x[1:1]);
y, formulation = MathOptAI.add_predictor(model, predictor, x);
y
formulation

```

## Reduced-space

Use the `reduced_space = true` keyword to formulate a reduced-space model:

```@repl
using JuMP, Lux, MathOptAI, Random
rng = Random.MersenneTwister();
chain = Lux.Chain(Lux.Dense(1 => 2, Lux.relu), Lux.Dense(2 => 1))
parameters, state = Lux.setup(rng, chain);
predictor = (chain, parameters, state);
model = Model();
@variable(model, x[1:1]);
y, formulation =
    MathOptAI.add_predictor(model, predictor, x; reduced_space = true);
y
formulation
```

## Gray-box

The Lux extension does not yet support the `gray_box` keyword argument.

## Change how layers are formulated

Pass a dictionary to the `config` keyword that maps Lux activation functions to
a MathOptAI predictor:

```@repl
using JuMP, Lux, MathOptAI, Random
rng = Random.MersenneTwister();
chain = Lux.Chain(Lux.Dense(1 => 2, Lux.relu), Lux.Dense(2 => 1))
parameters, state = Lux.setup(rng, chain);
predictor = (chain, parameters, state);
model = Model();
@variable(model, x[1:1]);
y, formulation = MathOptAI.add_predictor(
    model,
    predictor,
    x;
    config = Dict(Lux.relu => MathOptAI.ReLUSOS1()),
);
y
formulation
```
