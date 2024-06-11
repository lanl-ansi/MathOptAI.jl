```@meta
CurrentModule = MathOptAI
```

# API Reference

This page lists the public API of `MathOptAI`.

!!! info
    This page is an unstructured list of the MathOptAI API. For a more
    structured overview, read the Manual or Tutorial parts of this
    documentation.

Load all of the public the API into the current scope with:
```julia
using MathOptAI
```
Alternatively, load only the module with:
```julia
import MathOptAI
```
and then prefix all calls with `MathOptAI.` to create `MathOptAI.<NAME>`.

## `AbstractPredictor`
```@docs
AbstractPredictor
```

## `add_predictor`
```@docs
add_predictor
```

## `Affine`
```@docs
Affine
```

## `Pipeline`
```@docs
Pipeline
```

## `ReLU`
```@docs
ReLU
```

## `ReLUBigM`
```@docs
ReLUBigM
```

## `ReLUQuadratic`
```@docs
ReLUQuadratic
```

## `ReLUSOS1`
```@docs
ReLUSOS1
```

## `Sigmoid`
```@docs
Sigmoid
```

## `SoftMax`
```@docs
SoftMax
```

## `SoftPlus`
```@docs
SoftPlus
```

## `Tanh`
```@docs
Tanh
```

## `add_constraint`
```@docs
add_constraint
```

## `UnivariateNormalDistribution`
```@docs
UnivariateNormalDistribution
```
