# Predictors

The main entry point for embedding prediction models into JuMP is
[`add_predictor`](@ref).

All methods use the form `y = MathOptAI.add_predictor(model, predictor, x)` to
add the relationship `y = predictor(x)` to `model`.

## Supported predictors

The following predictors are supported. See their docstrings for details:

| Predictor          | Relationship                           |
| :----------------- | :------------------------------------- |
| [`Affine`](@ref)   |  $f(x) = Ax + b$                       |
| [`BinaryDecisionTree`](@ref) | A binary decision tree       |
| [`Pipeline`](@ref) |  $f(x) = (l_1 \circ \ldots \circ l_N)(x)$ |
| [`ReLU`](@ref)     |  $f(x) = \max.(0, x)$                  |
| [`ReLUBigM`](@ref) |  $f(x) = \max.(0, x)$                  |
| [`ReLUQuadratic`](@ref) |  $f(x) = \max.(0, x)$             |
| [`ReLUSOS1`](@ref) |  $f(x) = \max.(0, x)$                  |
| [`Sigmoid`](@ref)  |  $f(x) = \frac{1}{1 + e^{-x}}$         |
| [`SoftMax`](@ref)  |  $f(x) = \frac{e^{x_i}}{\sum e^{x_i}}$ |
| [`SoftPlus`](@ref) |  $f(x) = \log(1 + e^x)$                |
| [`Tanh`](@ref)     |  $f(x) = \tanh.(x)$                    |

Note that some predictors, such as the ReLU ones, offer multiple formulations of
the same mathematical relationship. The ''right'' choice is solver- and
problem-dependent.
