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

## `output_size`
```@docs
output_size
```

## `Affine`
```@docs
Affine
```

## `AffineCombination`
```@docs
AffineCombination
```

## `AvgPool2d`
```@docs
AvgPool2d
```

## `BinaryDecisionTree`
```@docs
BinaryDecisionTree
```

## `Conv2d`
```@docs
Conv2d
```

## `GELU`
```@docs
GELU
```

## `GrayBox`
```@docs
GrayBox
```

## `LeakyReLU`
```@docs
LeakyReLU
```

## `MaxPool2d`
```@docs
MaxPool2d
```

## `Permutation`
```@docs
Permutation
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

## `AbstractGPs`
```@autodocs
Modules = [Base.get_extension(MathOptAI, :MathOptAIAbstractGPsExt)]
```

## `DecisionTree`
```@autodocs
Modules = [Base.get_extension(MathOptAI, :MathOptAIDecisionTreeExt)]
```

## `EvoTrees`
```@autodocs
Modules = [Base.get_extension(MathOptAI, :MathOptAIEvoTreesExt)]
```

## `Flux`
```@autodocs
Modules = [Base.get_extension(MathOptAI, :MathOptAIFluxExt)]
```

## `GLM`
```@autodocs
Modules = [Base.get_extension(MathOptAI, :MathOptAIGLMExt)]
```

## `Lux`
```@autodocs
Modules = [Base.get_extension(MathOptAI, :MathOptAILuxExt)]
```

## `PythonCall`
```@autodocs
Modules = [Base.get_extension(MathOptAI, :MathOptAIPythonCallExt)]
```

## `StatsModels`
```@autodocs
Modules = [Base.get_extension(MathOptAI, :MathOptAIStatsModelsExt)]
```

## Extensions

```@docs
add_variables
get_variable_bounds
set_variable_bounds
get_variable_start
set_variable_start
```

## `replace_weights_with_variables`
```@docs
replace_weights_with_variables
```
