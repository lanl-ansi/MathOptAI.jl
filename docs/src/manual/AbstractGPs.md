# AbstractGPs

[AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl) is a
library for fittinng Gaussian Processes in Julia.

Here is an example:

```jldoctest
julia> using JuMP, MathOptAI, AbstractGPs

julia> x_data = 2π .* (0.0:0.1:1.0);

julia> y_data = sin.(x_data);

julia> fx = AbstractGPs.GP(AbstractGPs.Matern32Kernel())(x_data, 0.1);

julia> p_fx = AbstractGPs.posterior(fx, y_data);

julia> model = Model();

julia> @variable(model, 1 <= x[1:1] <= 6, start = 3);

julia> predictor = MathOptAI.Quantile(p_fx, [0.1, 0.9]);

julia> y = MathOptAI.add_predictor(model, predictor, x)
2-element Vector{VariableRef}:
 moai_quantile[1]
 moai_quantile[2]

julia> @objective(model, Max, y[2] - y[1])
moai_quantile[2] - moai_quantile[1]
```
