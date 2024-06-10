# Lux

[Lux.jl](https://github.com/LuxDL/Lux.jl) is a library for machine learning in
Julia.

MathOptAI supports embedding Lux models in JuMP if they are a chain composed
of:

 * `Lux.Dense`
 * `Lux.relu`
 * `Lux.sigmoid`
 * `Lux.softplus`
 * `Lux.tanh`

Here is an example:

```jldoctest
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
 moai_Affine[1]
```
