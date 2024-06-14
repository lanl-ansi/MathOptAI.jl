# Predictors

The main entry point for embedding prediction models into JuMP is
[`add_predictor`](@ref).

All methods use the form `y = MathOptAI.add_predictor(model, predictor, x)` to
add the relationship `y = predictor(x)` to `model`.

## Supported predictors

The following predictors are supported. See their docstrings for details:

| Predictor          | Relationship                           | Dimensions |
| :----------------- | :------------------------------------- | :--------- |
| [`Affine`](@ref)   |  $f(x) = Ax + b$                       | $M \rightarrow N$   |
| [`BinaryDecisionTree`](@ref) | A binary decision tree       | $M \rightarrow 1$   |
| [`Pipeline`](@ref) |  $f(x) = (l_1 \circ \ldots \circ l_N)(x)$ | $M \rightarrow N$ |
| [`ReLU`](@ref)     |  $f(x) = \max.(0, x)$                  | $M \rightarrow M$   |
| [`ReLUBigM`](@ref) |  $f(x) = \max.(0, x)$                  | $M \rightarrow M$   |
| [`ReLUQuadratic`](@ref) |  $f(x) = \max.(0, x)$             | $M \rightarrow M$   |
| [`ReLUSOS1`](@ref) |  $f(x) = \max.(0, x)$                  | $M \rightarrow M$   |
| [`Sigmoid`](@ref)  |  $f(x) = \frac{1}{1 + e^{-x}}$         | $M \rightarrow M$   |
| [`SoftMax`](@ref)  |  $f(x) = \frac{e^{x_i}}{\sum e^{x_i}}$ | $M \rightarrow 1$   |
| [`SoftPlus`](@ref) |  $f(x) = \log(1 + e^x)$                | $M \rightarrow M$   |
| [`Tanh`](@ref)     |  $f(x) = \tanh.(x)$                    | $M \rightarrow M$   |

Note that some predictors, such as the ReLU ones, offer multiple formulations of
the same mathematical relationship. The ''right'' choice is solver- and
problem-dependent.
