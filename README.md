![](https://upload.wikimedia.org/wikipedia/commons/2/22/Standing_Moai_at_Ahu_Tongariki%2C_Easter_Island%2C_Pacific_Ocean.jpg)

# MathOptAI.jl

[![Build Status](https://github.com/lanl-ansi/MathOptAI.jl/workflows/CI/badge.svg)](https://github.com/lanl-ansi/MathOptAI.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/lanl-ansi/MathOptAI.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/lanl-ansi/MathOptAI.jl)

MathOptAI.jl is a [JuMP](https://jump.dev) extension for embedding trained AI,
machine learning, and statistical learning models into a JuMP optimization
model.

## License

MathOptAI.jl is provided under a BSD-3 license as part of the Optimization and
Machine Learning Toolbox project, O4806.

See [LICENSE.md](https://github.com/lanl-ansi/MathOptAI.jl/blob/main/LICENSE.md)
for details.

_Despite the name similarity, this project is not affiliated with
[OMLT](https://github.com/cog-imperial/OMLT), the Optimization and Machine
Learning Toolkit._

## Installation

Install `MathOptAI.jl` using the Julia package manager:
```julia
import Pkg
Pkg.add("MathOptAI")
```

## Getting started

Here's an example of using MathOptAI to embed a trained neural network from Flux
into a JuMP model. The vector of JuMP variables `x` is fed as input to the
neural network. The output `y` is a vector of JuMP variables that represents the
output layer of the neural network. The `formulation` object stores the
additional variables and constraints that were added to `model`.

```julia
julia> using JuMP, MathOptAI, Flux

julia> predictor = Flux.Chain(
           Flux.Dense(28^2 => 32, Flux.sigmoid),
           Flux.Dense(32 => 10),
           Flux.softmax,
       );

julia> #= Train the Flux model. Code not shown for simplicity =#

julia> model = JuMP.Model();

julia> JuMP.@variable(model, 0 <= x[1:28^2] <= 1);

julia> y, formulation = MathOptAI.add_predictor(model, predictor, x);

julia> y
10-element Vector{VariableRef}:
 moai_SoftMax[1]
 moai_SoftMax[2]
 moai_SoftMax[3]
 moai_SoftMax[4]
 moai_SoftMax[5]
 moai_SoftMax[6]
 moai_SoftMax[7]
 moai_SoftMax[8]
 moai_SoftMax[9]
 moai_SoftMax[10]
```

## Documentation

Documentation is available at
[https://lanl-ansi.github.io/MathOptAI.jl/dev](https://lanl-ansi.github.io/MathOptAI.jl).

## Getting help

For help, questions, comments, and suggestions, please [open a GitHub issue](https://github.com/lanl-ansi/MathOptAI.jl/issues/new).

## Inspiration

This project is mainly inspired by two existing projects:

 * [OMLT](https://github.com/cog-imperial/OMLT)
 * [gurobi-machinelearning](https://github.com/Gurobi/gurobi-machinelearning)

Other works, from which we took less inspiration, include:

 * [JANOS](https://github.com/INFORMSJoC/2020.1023)
 * [MeLOn](https://git.rwth-aachen.de/avt-svt/public/MeLOn)
 * [ENTMOOT](https://github.com/cog-imperial/entmoot)
 * [reluMIP](https://github.com/process-intelligence-research/ReLU_ANN_MILP)
 * [OptiCL](https://github.com/hwiberg/OptiCL)
 * [PySCIPOpt-ML](https://github.com/Opt-Mucca/PySCIPOpt-ML)

The 2024 paper of López-Flores et al. is an excellent summary of the state of
the field at the time that we started development of MathOptAI.

> López-Flores, F.J., Ramírez-Márquez, C., Ponce-Ortega J.M. (2024). Process
> Systems Engineering Tools for Optimization of Trained Machine Learning Models:
> Comparative and Perspective. _Industrial & Engineering Chemistry Research_,
> 63(32), 13966-13979. DOI: [10.1021/acs.iecr.4c00632](https://pubs.acs.org/doi/abs/10.1021/acs.iecr.4c00632)
