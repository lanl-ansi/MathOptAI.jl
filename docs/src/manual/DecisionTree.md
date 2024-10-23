# DecisionTree

[DecisionTree.jl](https://github.com/JuliaAI/DecisionTree.jl) is a library for
fitting decision trees in Julia.

Here is an example:

```jldoctest
julia> using JuMP, MathOptAI, DecisionTree

julia> truth(x::Vector) = x[1] <= 0.5 ? -2 : (x[2] <= 0.3 ? 3 : 4)
truth (generic function with 1 method)

julia> features = abs.(sin.((1:10) .* (3:4)'));

julia> size(features)
(10, 2)

julia> labels = truth.(Vector.(eachrow(features)));

julia> predictor = DecisionTree.build_tree(labels, features)
Decision Tree
Leaves: 3
Depth:  2

julia> model = Model();

julia> @variable(model, 0 <= x[1:2] <= 1);

julia> y, formulation = MathOptAI.add_predictor(model, predictor, x);

julia> y
1-element Vector{VariableRef}:
 moai_BinaryDecisionTree_value
```
