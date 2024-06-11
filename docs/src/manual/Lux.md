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
julia> using JuMP, Lux, MathOptAI, Random

julia> rng = Random.MersenneTwister();

julia> chain = Lux.Chain(Lux.Dense(1 => 16, Lux.relu), Lux.Dense(16 => 1))
Chain(
    layer_1 = Dense(1 => 16, relu),     # 32 parameters
    layer_2 = Dense(16 => 1),           # 17 parameters
)         # Total: 49 parameters,
          #        plus 0 states.

julia> parameters, state = Lux.setup(rng, chain);

julia> predictor = (chain, parameters, state);

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
