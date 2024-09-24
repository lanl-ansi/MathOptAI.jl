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

| Predictor          | Relationship                           | Dimensions |
| :----------------- | :------------------------------------- | :--------- |
| [`Affine`](@ref)   |  $f(x) = Ax + b$                       | $M \rightarrow N$   |
| [`BinaryDecisionTree`](@ref) | A binary decision tree       | $M \rightarrow 1$   |
| [`GrayBox`](@ref)  |  $f(x)$                                | $M \rightarrow N$   |
| [`Pipeline`](@ref) |  $f(x) = (l_1 \circ \ldots \circ l_N)(x)$ | $M \rightarrow N$ |
| [`Quantile`](@ref) |  The quantiles of a distribution       | $M \rightarrow N$   |
| [`ReLU`](@ref)     |  $f(x) = \max.(0, x)$                  | $M \rightarrow M$   |
| [`ReLUBigM`](@ref) |  $f(x) = \max.(0, x)$                  | $M \rightarrow M$   |
| [`ReLUQuadratic`](@ref) |  $f(x) = \max.(0, x)$             | $M \rightarrow M$   |
| [`ReLUSOS1`](@ref) |  $f(x) = \max.(0, x)$                  | $M \rightarrow M$   |
| [`Scale`](@ref)    |  $f(x) = scale .* x .+ bias$           | $M \rightarrow M$   |
| [`Sigmoid`](@ref)  |  $f(x) = \frac{1}{1 + e^{-x}}$         | $M \rightarrow M$   |
| [`SoftMax`](@ref)  |  $f(x) = \frac{e^{x_i}}{\sum e^{x_i}}$ | $M \rightarrow 1$   |
| [`SoftPlus`](@ref) |  $f(x) = \frac{1}{\beta} \log(1 + e^{\beta x})$ | $M \rightarrow M$ |
| [`Tanh`](@ref)     |  $f(x) = \tanh.(x)$                    | $M \rightarrow M$   |

Note that some predictors, such as the ReLU ones, offer multiple formulations of
the same mathematical relationship. The ''right'' choice is solver- and
problem-dependent.

## ReLU

There are a number of different mathematical formulations for the rectified
linear unit (ReLU).

 * [`ReLU`](@ref): requires the solver to support the `max` nonlinear operator.
 * [`ReLUBigM`](@ref): requires the solver to support mixed-integer linear
   programs, and requires the user to have a priori knowledge of a suitable
   value for the "big-M" parameter.
 * [`ReLUQuadratic`](@ref): requires the solver to support quadratic equality
   constraints
 * [`ReLUSOS1`](@ref): requires the solver to support SOS-I constraints.

The correct choice for which ReLU formulation to use is problem- and
solver-dependent.
