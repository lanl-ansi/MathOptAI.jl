# ExaModels.jl

[ExaModels.jl](https://github.com/exanauts/ExaModels.jl) is an algebraic
modeling and automatic differentiation tool in Julia Language, specialized for
SIMD abstraction of nonlinear programs.

The upstream documentation is available at
[https://exanauts.github.io/ExaModels.jl/stable/](https://exanauts.github.io/ExaModels.jl/stable/).

## Supported layers

ExaModels supports the following predictors:

 * [`Affine`](@ref)
 * [`GELU`](@ref)
 * [`GrayBox`](@ref)
 * [`LeakyReLU`](@ref)
 * [`Permutation`](@ref)
 * [`Pipeline`](@ref)
 * [`ReLU`](@ref)
 * [`ReLUEpigraph`](@ref)
 * [`Scale`](@ref)
 * [`Sigmoid`](@ref)
 * [`SoftMax`](@ref)
 * [`SoftPlus`](@ref)
 * [`Tanh`](@ref)

## Basic example

Use [`MathOptAI.add_predictor`](@ref) to embed various predictors into an
`ExaCore`:

```@repl
using ExaModels, MathOptAI, Flux
chain = Flux.Chain(
    Flux.Dense(2 => 2, Flux.relu),
    Flux.Scale(2),
    Flux.Dense(2 => 2, Flux.sigmoid),
    Flux.softmax,
    Flux.Dense(2 => 2, Flux.softplus),
    Flux.Dense(2 => 2, Flux.tanh),
);
core = ExaModels.ExaCore(; concrete = Val(true))
core, x = ExaModels.add_var(core, 2)
(core, y), _ = MathOptAI.add_predictor(core, chain, x);
y
core
```

## Gray-box

Use the `gray_box = true` keyword to embed the network as a vector nonlinear
operator:

```@repl
using ExaModels, MathOptAI, Flux
chain = Flux.Chain(
    Flux.Dense(2 => 2, Flux.relu),
    Flux.Scale(2),
);
core = ExaModels.ExaCore(; concrete = Val(true))
core, x = ExaModels.add_var(core, 2)
(core, y), _ = MathOptAI.add_predictor(core, chain, x; gray_box = true);
y
core
```

## Change how layers are formulated

Pass a dictionary to the `config` keyword that maps Flux activation functions to
a MathOptAI predictor:

```@repl
using ExaModels, Flux, MathOptAI
predictor = Flux.Chain(Flux.Dense(1 => 2, Flux.relu), Flux.Dense(2 => 1));
core = ExaModels.ExaCore(; concrete = Val(true))
core, x = ExaModels.add_var(core, 2)
(core, y), _ = MathOptAI.add_predictor(
    core,
    predictor,
    x;
    config = Dict(Flux.relu => MathOptAI.ReLUEpigraph),
);
y
core
```
