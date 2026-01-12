# Lux.jl

[Lux.jl](https://github.com/LuxDL/Lux.jl) is a library for machine learning in
Julia.

The upstream documentation is available at
[https://lux.csail.mit.edu/stable/](https://lux.csail.mit.edu/stable/).

## Supported layers

MathOptAI supports embedding a Lux model into JuMP if it is a
[`Lux.Chain`](https://lux.csail.mit.edu/stable/api/Lux/layers#Lux.Chain)
composed of:

 * [`Lux.Dense`](https://lux.csail.mit.edu/stable/api/Lux/layers#Lux.Dense)
 * [`Lux.Scale`](https://lux.csail.mit.edu/stable/api/Lux/layers#Lux.Scale)
 * [`Lux.relu`](https://fluxml.ai/NNlib.jl/stable/reference/#NNlib.relu)
 * [`Lux.sigmoid`](https://fluxml.ai/NNlib.jl/stable/reference/#NNlib.sigmoid)
 * [`Lux.softmax`](https://fluxml.ai/NNlib.jl/stable/reference/#NNlib.softmax)
 * [`Lux.softplus`](https://fluxml.ai/NNlib.jl/stable/reference/#NNlib.softplus)
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
    config = Dict(Lux.relu => MathOptAI.ReLUSOS1),
);
y
formulation
```
