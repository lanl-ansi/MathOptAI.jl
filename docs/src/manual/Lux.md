# Lux

[Lux.jl](https://github.com/LuxDL/Lux.jl) is a library for machine learning in
Julia.

## Supported layers

MathOptAI supports embedding Lux models in JuMP if they are a chain composed
of:

 * `Lux.Dense`
 * `Lux.relu`
 * `Lux.sigmoid`
 * `Lux.softplus`
 * `Lux.tanh`

## Basic example

Use [`MathOptAI.add_predictor`](@ref) to embed a tuple (containing the
`Lux.Chain`, the `parameters`, and the `state`) into a JuMP model:

```jldoctest
julia> using JuMP, Lux, MathOptAI, Random

julia> rng = Random.MersenneTwister();

julia> chain = Lux.Chain(Lux.Dense(1 => 2, Lux.relu), Lux.Dense(2 => 1))
Chain(
    layer_1 = Dense(1 => 2, relu),      # 4 parameters
    layer_2 = Dense(2 => 1),            # 3 parameters
)         # Total: 7 parameters,
          #        plus 0 states.

julia> parameters, state = Lux.setup(rng, chain);

julia> predictor = (chain, parameters, state);

julia> model = Model();

julia> @variable(model, x[1:1]);

julia> y, formulation = MathOptAI.add_predictor(model, predictor, x);

julia> y
1-element Vector{VariableRef}:
 moai_Affine[1]

julia> formulation
Affine(A, b) [input: 1, output: 2]
├ variables [2]
│ ├ moai_Affine[1]
│ └ moai_Affine[2]
└ constraints [2]
  ├ -1.3048839569091797 x[1] - moai_Affine[1] = 0
  └ 0.9908649921417236 x[1] - moai_Affine[2] = 0
ReLU()
├ variables [2]
│ ├ moai_ReLU[1]
│ └ moai_ReLU[2]
└ constraints [4]
  ├ moai_ReLU[1] ≥ 0
  ├ moai_ReLU[2] ≥ 0
  ├ moai_ReLU[1] - max(0.0, moai_Affine[1]) = 0
  └ moai_ReLU[2] - max(0.0, moai_Affine[2]) = 0
Affine(A, b) [input: 2, output: 1]
├ variables [1]
│ └ moai_Affine[1]
└ constraints [1]
  └ -0.47718897461891174 moai_ReLU[1] + 0.9381989240646362 moai_ReLU[2] - moai_Affine[1] = 0
```

## Reduced-space

Use the `reduced_space = true` keyword to formulate a reduced-space model:

```jldoctest
julia> using JuMP, Lux, MathOptAI, Random

julia> rng = Random.MersenneTwister();

julia> chain = Lux.Chain(Lux.Dense(1 => 2, Lux.relu), Lux.Dense(2 => 1))
Chain(
    layer_1 = Dense(1 => 2, relu),      # 4 parameters
    layer_2 = Dense(2 => 1),            # 3 parameters
)         # Total: 7 parameters,
          #        plus 0 states.

julia> parameters, state = Lux.setup(rng, chain);

julia> predictor = (chain, parameters, state);

julia> model = Model();

julia> @variable(model, x[1:1]);

julia> y, formulation =
           MathOptAI.add_predictor(model, predictor, x; reduced_space = true);

julia> y
1-element Vector{NonlinearExpr}:
 (+(0.0) + (1.159173846244812 * max(0.0, -1.1450185775756836 x[1]))) + (1.410927414894104 * max(0.0, -0.9511919617652893 x[1])) + 0.0

julia> formulation
ReducedSpace(Affine(A, b) [input: 1, output: 2])
├ variables [0]
└ constraints [0]
ReducedSpace(ReLU())
├ variables [0]
└ constraints [0]
ReducedSpace(Affine(A, b) [input: 2, output: 1])
├ variables [0]
└ constraints [0]
```

## Gray-box

The Lux extension does not yet support the `gray_box` keyword argument.

## Change how layers are formulated

Pass a dictionary to the `config` keyword that maps Lux activation functions to
a MathOptAI predictor:

```jldoctest
julia> using JuMP, Lux, MathOptAI, Random

julia> rng = Random.MersenneTwister();

julia> chain = Lux.Chain(Lux.Dense(1 => 2, Lux.relu), Lux.Dense(2 => 1))
Chain(
    layer_1 = Dense(1 => 2, relu),      # 4 parameters
    layer_2 = Dense(2 => 1),            # 3 parameters
)         # Total: 7 parameters,
          #        plus 0 states.

julia> parameters, state = Lux.setup(rng, chain);

julia> predictor = (chain, parameters, state);

julia> model = Model();

julia> @variable(model, x[1:1]);

julia> y, formulation = MathOptAI.add_predictor(
           model,
           predictor,
           x;
           config = Dict(Lux.relu => MathOptAI.ReLUSOS1()),
       );

julia> y
1-element Vector{VariableRef}:
 moai_Affine[1]

julia> formulation
Affine(A, b) [input: 1, output: 2]
├ variables [2]
│ ├ moai_Affine[1]
│ └ moai_Affine[2]
└ constraints [2]
  ├ -1.3204172849655151 x[1] - moai_Affine[1] = 0
  └ 1.1320137977600098 x[1] - moai_Affine[2] = 0
ReLUSOS1()
├ variables [4]
│ ├ moai_ReLU[1]
│ ├ moai_ReLU[2]
│ ├ moai_z[1]
│ └ moai_z[2]
└ constraints [4]
  ├ moai_Affine[1] - moai_ReLU[1] + moai_z[1] = 0
  ├ moai_Affine[2] - moai_ReLU[2] + moai_z[2] = 0
  ├ [moai_ReLU[1], moai_z[1]] ∈ MathOptInterface.SOS1{Float64}([1.0, 2.0])
  └ [moai_ReLU[2], moai_z[2]] ∈ MathOptInterface.SOS1{Float64}([1.0, 2.0])
Affine(A, b) [input: 2, output: 1]
├ variables [1]
│ └ moai_Affine[1]
└ constraints [1]
  └ 0.6522457599639893 moai_ReLU[1] - 1.3529249429702759 moai_ReLU[2] - moai_Affine[1] = 0
```
