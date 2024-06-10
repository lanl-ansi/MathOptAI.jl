# Flux

[Flux.jl](https://github.com/FluxML/Flux.jl) is a library for machine learning
in Julia.

MathOptAI supports embedding Flux models in JuMP if they are a chain composed
of:

 * `Flux.Dense`
 * `Flux.softmax`
 * `Flux.relu`
 * `Flux.sigmoid`
 * `Flux.softplus`
 * `Flux.tanh`

Here is an example:

```jldoctest
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
 moai_Affine[1]
```
