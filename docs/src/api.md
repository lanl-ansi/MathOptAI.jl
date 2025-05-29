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

## `build_predictor`
```@docs
build_predictor(predictor::AbstractPredictor; kwargs...)
```

## `Affine`
```@docs
Affine
```

## `BinaryDecisionTree`
```@docs
BinaryDecisionTree
```

## `GrayBox`
```@docs
GrayBox
```

## `LinearCombination`
```@docs
LinearCombination
```

## `Pipeline`
```@docs
Pipeline
```

## `PytorchModel`
```@docs
PytorchModel
```

## `Quantile`
```@docs
Quantile
```

## `ReducedSpace`
```@docs
ReducedSpace
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

## `Scale`
```@docs
Scale
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

## `VectorNonlinearOracle`
```@docs
VectorNonlinearOracle
```

## `AbstractFormulation`
```@docs
AbstractFormulation
```

## `Formulation`
```@docs
Formulation
```

## `PipelineFormulation`
```@docs
PipelineFormulation
```

## Extensions

```@autodocs
Modules = [
    Base.get_extension(MathOptAI, :MathOptAIAbstractGPsExt),
    Base.get_extension(MathOptAI, :MathOptAIDecisionTreeExt),
    Base.get_extension(MathOptAI, :MathOptAIFluxExt),
    Base.get_extension(MathOptAI, :MathOptAIGLMExt),
    Base.get_extension(MathOptAI, :MathOptAILuxExt),
    Base.get_extension(MathOptAI, :MathOptAIPythonCallExt),
    Base.get_extension(MathOptAI, :MathOptAIStatsModelsExt),
]
```
