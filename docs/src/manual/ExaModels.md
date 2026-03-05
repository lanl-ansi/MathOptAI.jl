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
 * [`LeakyReLU`](@ref)
 * [`Permutation`](@ref)
 * [`Pipeline`](@ref)
 * [`ReLU`](@ref)
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
    Flux.Dense(2 => 2, Flux.softmax),
    Flux.Dense(2 => 2, Flux.softplus),
    Flux.Dense(2 => 2, Flux.tanh),
);
core = ExaModels.ExaCore()
x = ExaModels.variable(core, 2)
y, _ = MathOptAI.add_predictor(core, chain, x);
y
core
```
