# GLM

[GLM.jl](https://github.com/JuliaStats/GLM.jl) is a library for fitting
generalized linear models in Julia.

MathOptAI.jl supports embedding two types of regression models from GLM:

 * `GLM.lm(X, Y)`
 * `GLM.glm(X, Y, GLM.Bernoulli())`

## Linear regression

The input `x` to [`add_predictor`](@ref) must be a vector with the same number
of elements as columns in the training matrix. The return is a vector of JuMP
variables with a single element.

```@repl
using GLM, JuMP, MathOptAI
X, Y = rand(10, 2), rand(10);
predictor = GLM.lm(X, Y);
model = Model();
@variable(model, x[1:2]);
y, formulation = MathOptAI.add_predictor(model, predictor, x);
y
formulation
```

## Logistic regression

The input `x` to [`add_predictor`](@ref) must be a vector with the same number
of elements as columns in the training matrix. The return is a vector of JuMP
variables with a single element.

```@repl
using GLM, JuMP, MathOptAI
X, Y = rand(10, 2), rand(Bool, 10);
predictor = GLM.glm(X, Y, GLM.Bernoulli());
model = Model();
@variable(model, x[1:2]);
y, formulation = MathOptAI.add_predictor(model, predictor, x);
y
formulation
```

## DataFrames

[DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) can be used with
GLM.jl.

The input `x` to [`add_predictor`](@ref) must be a DataFrame with the same
feature columns as the training DataFrame. The return is a vector of JuMP
variables, with one element for each row in the DataFrame.

```@repl
using DataFrames, GLM, JuMP, MathOptAI
train_df = DataFrames.DataFrame(x1 = rand(10), x2 = rand(10));
train_df.y = 1.0 .* train_df.x1 + 2.0 .* train_df.x2 .+ rand(10);
predictor = GLM.lm(GLM.@formula(y ~ x1 + x2), train_df);
model = Model();
test_df = DataFrames.DataFrame(
    x1 = rand(6),
    x2 = @variable(model, [1:6]),
);
test_df.y, _ = MathOptAI.add_predictor(model, predictor, test_df);
test_df.y
```

