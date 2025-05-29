# DecisionTree.jl

[DecisionTree.jl](https://github.com/JuliaAI/DecisionTree.jl) is a library for
fitting decision trees in Julia.

## Binary decision tree regression

Here is an example:

```@repl
using JuMP, MathOptAI, DecisionTree
truth(x::Vector) = x[1] <= 0.5 ? -2 : (x[2] <= 0.3 ? 3 : 4)
features = abs.(sin.((1:10) .* (3:4)'));
size(features)
labels = truth.(Vector.(eachrow(features)));
predictor = DecisionTree.build_tree(labels, features)
model = Model();
@variable(model, 0 <= x[1:2] <= 1);
y, formulation = MathOptAI.add_predictor(model, predictor, x);
y
formulation
```

## Random forest regression

```@repl
using JuMP, MathOptAI, DecisionTree
truth(x::Vector) = x[1] <= 0.5 ? -2 : (x[2] <= 0.3 ? 3 : 4)
features = abs.(sin.((1:10) .* (3:4)'));
size(features)
labels = truth.(Vector.(eachrow(features)));
predictor = DecisionTree.build_forest(labels, features)
model = Model();
@variable(model, 0 <= x[1:2] <= 1);
y, formulation = MathOptAI.add_predictor(model, predictor, x);
y
formulation
```
