# Omelette

_If you can come up with a better name, please open an issue._

Omelette is a [JuMP](https://jump.dev) extension for embedding common types of
AI, machine learning, and statistical learning models into a JuMP optimization
model.

## License

Omelette.jl is licensed under the [MIT license](https://github.com/lanl-ansi/jump-ml/blob/main/LICENSE.md)

## Getting help

This package is under active development. For help, questions, comments, and
suggestions, please open a GitHub issue.

## Inspiration

This project is inspired by two existing projects:

 * [OMLT](https://github.com/cog-imperial/OMLT)
 * [gurobi-machinelearning](https://github.com/Gurobi/gurobi-machinelearning)

## Supported models

Use `Omelette.add_predictor(model, predictor, x)` to add the relationship
`y = predictor(x)` to `model`:

```julia
y = Omelette.add_predictor(model, predictor, x)
```

The following predictors are supported. See their docstrings for details:

 * `Omelette.LinearRegression`
 * `Omelette.LogisticRegression`
 * `Omelette.Pipeline`
 * `Omelette.ReLU`
 * `Omelette.ReLUBigM`
 * `Omelette.ReLUQuadratic`
 * `Omelette.ReLUSOS1`
 * `Omelette.Sigmoid`
 * `Omelette.SoftPlus`
 * `Omelette.Tanh`

## Extensions

The following third-party package extensions are supported.

### [GLM.jl](https://github.com/JuliaStats/GLM.jl)

#### LinearRegression

```julia
using Omelette, GLM
X, Y = rand(10, 2), rand(10)
model_glm = GLM.lm(X, Y)
y = Omelette.add_predictor(model, model_glm, x)
```

#### LogisticRegression

```julia
using Omelette, GLM
X, Y = rand(10, 2), rand(Bool, 10)
model_glm = GLM.glm(X, Y, GLM.Bernoulli())
y = Omelette.add_predictor(model, model_glm, x)
```

### [Lux.jl](https://github.com/LuxDL/Lux.jl)

See `test/test_Lux.jl` for an example.

## Other constraints

### UnivariateNormalDistribution

```julia
using JuMP, Omelette
model = Model();
@variable(model, 0 <= x <= 5);
f = Omelette.UnivariateNormalDistribution(;
    mean = x -> only(x),
    covariance = x -> 1.0,
);
Omelette.add_constraint(model, f, [x], MOI.Interval(0.5, Inf), 0.95);
```
