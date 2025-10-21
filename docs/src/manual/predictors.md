```@meta
CurrentModule = MathOptAI
```

# Predictors

The main entry point for embedding prediction models into JuMP is
[`add_predictor`](@ref).

All methods use the form `y, formulation = MathOptAI.add_predictor(model, predictor, x)`
to add the relationship `y = predictor(x)` to `model`.

## Supported predictors

The following predictors are supported. See their docstrings for details:

| Predictor                       | Problem class        | Dimensions        |
| :------------------------------ | :------------------- | :---------------- |
| [`Affine`](@ref)                | Linear               | $M \rightarrow N$ |
| [`AffineCombination`](@ref)     | Linear               | $M \rightarrow N$ |
| [`BinaryDecisionTree`](@ref)    | Mixed-integer linear | $M \rightarrow 1$ |
| [`GrayBox`](@ref)               | Local nonlinear      | $M \rightarrow N$ |
| [`Pipeline`](@ref)              |                      | $M \rightarrow N$ |
| [`Quantile`](@ref)              | Local nonlinear      | $M \rightarrow N$ |
| [`ReLU`](@ref)                  | Global nonlinear     | $M \rightarrow M$ |
| [`ReLUBigM`](@ref)              | Mixed-integer linear | $M \rightarrow M$ |
| [`ReLUQuadratic`](@ref)         | Non-convex quadratic | $M \rightarrow M$ |
| [`ReLUSOS1`](@ref)              | Mixed-integer linear | $M \rightarrow M$ |
| [`Scale`](@ref)                 | Linear               | $M \rightarrow M$ |
| [`Sigmoid`](@ref)               | Global nonlinear     | $M \rightarrow M$ |
| [`SoftMax`](@ref)               | Global nonlinear     | $M \rightarrow M$ |
| [`SoftPlus`](@ref)              | Global nonlinear     | $M \rightarrow M$ |
| [`Tanh`](@ref)                  | Global nonlinear     | $M \rightarrow M$ |
