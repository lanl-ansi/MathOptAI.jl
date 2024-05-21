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

Use `add_model` to add a model.
```julia
Omelette.add_model(model, model_ml, x, y)
y = Omelette.add_model(model, model_ml, x)
```

### LinearRegression

```julia
num_features, num_observations = 2, 10
X = rand(num_observations, num_features)
θ = rand(num_features)
Y = X * θ + randn(num_observations)
model_glm = GLM.lm(X, Y)
model_ml = Omelette.LinearRegression(model_glm)
model = Model(HiGHS.Optimizer)
set_silent(model)
@variable(model, 0 <= x[1:num_features] <= 1)
@constraint(model, sum(x) == 1.5)
y = Omelette.add_model(model, model_ml, x)
@objective(model, Max, y[1])
```
