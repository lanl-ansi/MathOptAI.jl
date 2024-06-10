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

```jldoctest
julia> using GLM, JuMP, MathOptAI

julia> X, Y = rand(10, 2), rand(10);

julia> model_glm = GLM.lm(X, Y);

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> y = MathOptAI.add_predictor(model, model_glm, x)
1-element Vector{VariableRef}:
 moai_Affine[1]
```

## Logistic regression

The input `x` to [`add_predictor`](@ref) must be a vector with the same number
of elements as columns in the training matrix. The return is a vector of JuMP
variables with a single element.

```jldoctest
julia> using GLM, JuMP, MathOptAI

julia> X, Y = rand(10, 2), rand(Bool, 10);

julia> model_glm = GLM.glm(X, Y, GLM.Bernoulli());

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> y = MathOptAI.add_predictor(model, model_glm, x)
1-element Vector{VariableRef}:
 moai_Sigmoid[1]
```

## DataFrames

[DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) can be used with
GLM.jl.

The input `x` to [`add_predictor`](@ref) must be a DataFrame with the same
feature columns as the training DataFrame. The return is a vector of JuMP
variables, with one element for each row in the DataFrame.

```jldoctest
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
 moai_Affine[1]
 moai_Affine[1]
 moai_Affine[1]
 moai_Affine[1]
 moai_Affine[1]
 moai_Affine[1]
```

