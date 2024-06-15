import AbstractGPs
import Ipopt
import MathOptAI
import Plots
using JuMP

# Univariate example
f = x -> sin(x)
x = 2π .* (0.0:0.1:1.0)
y = f.(x)
fx = AbstractGPs.GP(AbstractGPs.Matern32Kernel())(x, 0.1)
p_fx = AbstractGPs.posterior(fx, y)

model = Model(Ipopt.Optimizer)
@variable(model, 1 <= x_input[1:1] <= 6, start = 3)
@objective(model, Max, x_input[1])
predictor = MathOptAI.Quantile(p_fx, [0.1, 0.9])
y_output = MathOptAI.add_predictor(model, predictor, x_input)
set_lower_bound(y_output[1], 0.0)  # P(p_fx(x) >= 0.0) >= 0.9
set_upper_bound(y_output[2], 0.5)  # P(p_fx(x) <= 0.5) >= 0.9
optimize!(model)
@assert is_solved_and_feasible(model)
Plots.plot(0:0.1:2π+0.1, f; label = "True function")
Plots.scatter!(x, y; label = "Data")
Plots.plot!(0:0.1:2π+0.1, p_fx; label = "GP")
Plots.hline!([0.5]; label = false)
Plots.vline!(value.(x_input); label = false)

# Univariate example
x = 2π .* rand(20, 2)
y = vec(sum(sin.(x); dims = 2))
kernel = AbstractGPs.Matern32Kernel()
fx = AbstractGPs.GP(kernel)(AbstractGPs.RowVecs(x), 0.01)
p_fx = AbstractGPs.posterior(fx, y)


model = Model(Ipopt.Optimizer)
@variable(model, 1 <= x_input[1:2] <= 6, start = 3)
@objective(model, Max, sum(x_input))
predictor = Quantile(p_fx, 0.9)
y_output = MathOptAI.add_predictor(model, predictor, x_input)
set_lower_bound(only(y_output), 0.5)
optimize!(model)
@assert is_solved_and_feasible(model)
value.(y_output)
value.(x)
