# Flux

[Flux.jl](https://github.com/FluxML/Flux.jl) is a library for machine learning
in Julia.

## Supported layers

MathOptAI supports embedding Flux models in JuMP if they are a `Flux.Chain`
composed of:

 * `Flux.Dense`
 * `Flux.softmax`
 * `Flux.relu`
 * `Flux.sigmoid`
 * `Flux.softplus`
 * `Flux.tanh`

## Basic example

Use [`MathOptAI.add_predictor`](@ref) to embed a `Flux.Chain` into a JuMP model:

```jldoctest
julia> using JuMP, Flux, MathOptAI

julia> chain = Flux.Chain(Flux.Dense(1 => 2, Flux.relu), Flux.Dense(2 => 1));

julia> model = Model();

julia> @variable(model, x[1:1]);

julia> y, formulation = MathOptAI.add_predictor(model, chain, x);

julia> y
1-element Vector{VariableRef}:
 moai_Affine[1]

julia> formulation
Affine(A, b) [input: 1, output: 2]
├ variables [2]
│ ├ moai_Affine[1]
│ └ moai_Affine[2]
└ constraints [2]
  ├ -1.140453577041626 x[1] - moai_Affine[1] = 0
  └ -0.3172467350959778 x[1] - moai_Affine[2] = 0
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
  └ -0.28468137979507446 moai_ReLU[1] - 1.0913821458816528 moai_ReLU[2] - moai_Affine[1] = 0
```

## Reduced-space

Use the `reduced_space = true` keyword to formulate a reduced-space model:

```jldoctest
julia> using JuMP, Flux, MathOptAI

julia> chain = Flux.Chain(Flux.Dense(1 => 2, Flux.relu), Flux.Dense(2 => 1));

julia> model = Model();

julia> @variable(model, x[1:1]);

julia> y, formulation =
           MathOptAI.add_predictor(model, chain, x; reduced_space = true);

julia> y
1-element Vector{NonlinearExpr}:
 (+(0.0) + (0.1822485327720642 * max(0.0, 0.5964527726173401 x[1]))) + (0.28217753767967224 * max(0.0, 0.05296982079744339 x[1])) + 0.0

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

Use the `gray_box = true` keyword to embed the network as a nonlinear operator:

```jldoctest
julia> using JuMP, Flux, MathOptAI

julia> chain = Flux.Chain(Flux.Dense(1 => 2, Flux.relu), Flux.Dense(2 => 1));

julia> model = Model();

julia> @variable(model, x[1:1]);

julia> y, formulation =
           MathOptAI.add_predictor(model, chain, x; gray_box = true);

julia> y
1-element Vector{VariableRef}:
 moai_GrayBox[1]

julia> formulation
GrayBox
├ variables [1]
│ └ moai_GrayBox[1]
└ constraints [1]
  └ op_##235(x[1]) - moai_GrayBox[1] = 0
```

## Change how layers are formulated

```jldoctest
julia> using JuMP, Flux, MathOptAI

julia> chain = Flux.Chain(Flux.Dense(1 => 2, Flux.relu), Flux.Dense(2 => 1));

julia> model = Model();

julia> @variable(model, x[1:1]);

julia> y, formulation = MathOptAI.add_predictor(
           model,
           chain,
           x;
           config = Dict(Flux.relu => MathOptAI.ReLUSOS1()),
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
  ├ 0.2895528972148895 x[1] - moai_Affine[1] = 0
  └ 0.932373046875 x[1] - moai_Affine[2] = 0
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
  └ -1.0097640752792358 moai_ReLU[1] - 0.5968713760375977 moai_ReLU[2] - moai_Affine[1] = 0
```
