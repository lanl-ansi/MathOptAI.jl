# EvoTrees.jl

[EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl) is a library for
fitting decision trees in Julia.

## Gradient boosted tree regression

Here is an example:

```@repl
using JuMP, MathOptAI, EvoTrees
truth(x::Vector) = x[1] <= 0.5 ? -2 : (x[2] <= 0.3 ? 3 : 4)
x_train = abs.(sin.((1:10) .* (3:4)'));
size(x_train)
y_train = truth.(Vector.(eachrow(x_train)));
config = EvoTrees.EvoTreeRegressor(; nrounds = 3);
predictor = EvoTrees.fit(config; x_train, y_train);
model = Model();
@variable(model, 0 <= x[1:2] <= 1);
y, formulation = MathOptAI.add_predictor(model, predictor, x);
y
formulation
```
