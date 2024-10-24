# AbstractGPs.jl

[AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl) is a
library for fittinng Gaussian Processes in Julia.

## Basic example

Here is an example:

```@repl
using JuMP, MathOptAI, AbstractGPs
x_data = 2π .* (0.0:0.1:1.0);
y_data = sin.(x_data);
fx = AbstractGPs.GP(AbstractGPs.Matern32Kernel())(x_data, 0.1);
p_fx = AbstractGPs.posterior(fx, y_data);
model = Model();
@variable(model, 1 <= x[1:1] <= 6, start = 3);
predictor = MathOptAI.Quantile(p_fx, [0.1, 0.9]);
y, formulation = MathOptAI.add_predictor(model, predictor, x);
y
formulation
@objective(model, Max, y[2] - y[1])
```
