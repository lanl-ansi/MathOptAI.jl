![](https://upload.wikimedia.org/wikipedia/commons/2/22/Standing_Moai_at_Ahu_Tongariki%2C_Easter_Island%2C_Pacific_Ocean.jpg)

# MathOptAI (Mo'ai)

_If you can come up with a better name, please open an issue._

MathOptAI.jl (Mo'ai) is a [JuMP](https://jump.dev) extension for embedding
trained AI, machine learning, and statistical learning models into a JuMP
optimization model.

## License

MathOptAI.jl is licensed under the [MIT license](https://github.com/lanl-ansi/jump-ml/blob/main/LICENSE.md)

## Getting help

This package is under active development. For help, questions, comments, and
suggestions, please open a GitHub issue.

## Inspiration

This project is inspired by two existing projects:

 * [OMLT](https://github.com/cog-imperial/OMLT)
 * [gurobi-machinelearning](https://github.com/Gurobi/gurobi-machinelearning)

## Predictors

Use `MathOptAI.add_predictor(model, predictor, x)` to add the relationship
`y = predictor(x)` to `model`:

```julia
y = MathOptAI.add_predictor(model, predictor, x)
```

The following predictors are supported. See their docstrings for details:

 * `MathOptAI.LinearRegression`
 * `MathOptAI.LogisticRegression`
 * `MathOptAI.Pipeline`
 * `MathOptAI.ReLU`
 * `MathOptAI.ReLUBigM`
 * `MathOptAI.ReLUQuadratic`
 * `MathOptAI.ReLUSOS1`
 * `MathOptAI.Sigmoid`
 * `MathOptAI.SoftPlus`
 * `MathOptAI.Tanh`

## Extensions

The following third-party package extensions are supported.

### [GLM.jl](https://github.com/JuliaStats/GLM.jl)

#### LinearRegression

```julia
using MathOptAI, GLM
X, Y = rand(10, 2), rand(10)
model_glm = GLM.lm(X, Y)
model = Model()
@variable(model, x[1:2])
y = MathOptAI.add_predictor(model, model_glm, x)
```

#### LogisticRegression

```julia
using JuMP, MathOptAI, GLM
X, Y = rand(10, 2), rand(Bool, 10)
model_glm = GLM.glm(X, Y, GLM.Bernoulli())
model = Model()
@variable(model, x[1:2])
y = MathOptAI.add_predictor(model, model_glm, x)
```

### [Lux.jl](https://github.com/LuxDL/Lux.jl)

See `test/test_Lux.jl` for an example.

## Constraints

### UnivariateNormalDistribution

```julia
using JuMP, MathOptAI
model = Model();
@variable(model, 0 <= x <= 5);
f = MathOptAI.UnivariateNormalDistribution(;
    mean = x -> only(x),
    covariance = x -> 1.0,
);
MathOptAI.add_constraint(model, f, [x], MOI.Interval(0.5, Inf), 0.95);
```
