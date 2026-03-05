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

| Predictor                    | Problem class         | JuMP | ExaModels |
| :--------------------------- | :-------------------- | :--- | :-------- |
| [`Affine`](@ref)             | Linear                | Yes  | Yes       |
| [`AffineCombination`](@ref)  | Linear                | Yes  |           |
| [`AvgPool2d`](@ref)          | Linear                | Yes  |           |
| [`BinaryDecisionTree`](@ref) | Mixed-integer linear  | Yes  |           |
| [`Conv2d`](@ref).            | Linear                | Yes  |           |
| [`GCNConv`](@ref)            | Linear                | Yes  |           |
| [`GELU`](@ref)               | Global nonlinear      | Yes  | Yes       |
| [`GrayBox`](@ref)            | Local nonlinear       | Yes  |           |
| [`LayerNorm`](@ref)          | Global nonlinear      | Yes  |           |
| [`LeakyReLU`](@ref)          | Depends on inner ReLU | Yes  | Yes       |
| [`MaxPool2d`](@ref)          | Global nonlinear      | Yes  |           |
| [`MaxPool2dBigM`](@ref)      | Mixed-integer linear  | Yes  |           |
| [`Pipeline`](@ref)           |                       | Yes  | Yes       |
| [`Quantile`](@ref)           | Local nonlinear       | Yes  |           |
| [`ReLU`](@ref)               | Global nonlinear      | Yes  | Yes       |
| [`ReLUBigM`](@ref)           | Mixed-integer linear  | Yes  |           |
| [`ReLUQuadratic`](@ref)      | Non-convex quadratic  | Yes  |           |
| [`ReLUSOS1`](@ref)           | Mixed-integer linear  | Yes  |           |
| [`Scale`](@ref)              | Linear                | Yes  | Yes       |
| [`Sigmoid`](@ref)            | Global nonlinear      | Yes  | Yes       |
| [`SoftMax`](@ref)            | Global nonlinear      | Yes  | Yes       |
| [`SoftPlus`](@ref)           | Global nonlinear      | Yes  | Yes       |
| [`TAGConv`](@ref)            | Linear                | Yes  |           |
| [`Tanh`](@ref)               | Global nonlinear      | Yes  | Yes       |
