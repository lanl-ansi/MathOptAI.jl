![](https://upload.wikimedia.org/wikipedia/commons/2/22/Standing_Moai_at_Ahu_Tongariki%2C_Easter_Island%2C_Pacific_Ocean.jpg)

# MathOptAI (Mo'ai)

_If you can come up with a better name, please open an issue._

MathOptAI.jl (Mo'ai) is a [JuMP](https://jump.dev) extension for embedding
trained AI, machine learning, and statistical learning models into a JuMP
optimization model.

## License

MathOptAI.jl is provided under a BSD license.

See [LICENSE.md](LICENSE.md) for details.

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

 * `MathOptAI.Affine`
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

#### Linear regression

```julia
julia> using GLM, JuMP, MathOptAI

julia> X, Y = rand(10, 2), rand(10);

julia> model_glm = GLM.lm(X, Y);

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> y = MathOptAI.add_predictor(model, model_glm, x)
1-element Vector{VariableRef}:
 omelette_Affine[1]
```

#### Logistic regression

```julia
julia> using GLM, JuMP, MathOptAI

julia> X, Y = rand(10, 2), rand(Bool, 10);

julia> model_glm = GLM.glm(X, Y, GLM.Bernoulli());

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> y = MathOptAI.add_predictor(model, model_glm, x)
1-element Vector{VariableRef}:
 omelette_Sigmoid[1]
```

### [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl)

DataFrames can be used with GLM.jl.

```julia
julia> using DataFrames, GLM, JuMP, MathOptAI

julia> train_df = DataFrames.DataFrame(x1 = rand(10), x2 = rand(10));

julia> train_df.y = 1.0 .* train_df.x1 + 2.0 .* train_df.x2 .+ rand(10);

julia> predictor = GLM.lm(GLM.@formula(y ~ x1 + x2), train_df);

julia> model = Model();

julia> test_df = DataFrames.DataFrame(
           x1 = rand(6),
           x2 = @variable(model, [1:6]),
       );

julia> test_df.y = MathOptAI.add_predictor(model, predictor, test_df)
6-element Vector{VariableRef}:
 omelette_Affine[1]
 omelette_Affine[1]
 omelette_Affine[1]
 omelette_Affine[1]
 omelette_Affine[1]
 omelette_Affine[1]
```

### [Flux.jl](https://github.com/FluxML/Flux.jl)

#### Neural networks

```julia
julia> using JuMP, Flux, MathOptAI

julia> chain = Flux.Chain(Flux.Dense(1 => 16, Flux.relu), Flux.Dense(16 => 1));

julia> model = Model();

julia> @variable(model, x[1:1]);

julia> y = MathOptAI.add_predictor(
           model,
           chain,
           x;
           config = Dict(Flux.relu => MathOptAI.ReLU()),
       )
1-element Vector{VariableRef}:
 omelette_Affine[1]
```

See [test/test_Flux.jl](test/test_Flux.jl) for more details.


### [Lux.jl](https://github.com/LuxDL/Lux.jl)

#### Neural networks

```julia
julia> using JuMP, Lux, MathOptAI, Random, Optimisers

julia> predictor = Lux.Experimental.TrainState(
           Random.MersenneTwister(),
           Lux.Chain(Lux.Dense(1 => 16, Lux.relu), Lux.Dense(16 => 1)),
           Optimisers.Adam(0.03f0),
       );

julia> model = Model();

julia> @variable(model, x[1:1]);

julia> y = MathOptAI.add_predictor(
           model,
           predictor,
           x;
           config = Dict(Lux.relu => MathOptAI.ReLU()),
       )
1-element Vector{VariableRef}:
 omelette_Affine[1]
```

See [test/test_Lux.jl](test/test_Lux.jl) for details.

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
