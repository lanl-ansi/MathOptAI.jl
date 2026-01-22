```@meta
CurrentModule = MathOptAI
```

# Release notes

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Version 0.2.2 (January 22, 2026)

### Added

 - Added support for `nn.Dropout` (#239)
 - Added [`LayerNorm`](@ref) predictor (#240)
 - Added [`GCNConv`](@ref) and [`TAGConv`](@ref) predictors (#244)

### Other

 - Updated printing in docstrings for JuMP@1.29.4 (#241)
 - Added checklists to docs (#242)

## Version 0.2.1 (January 15, 2026)

### Added

 - Added support for custom layers in PyTorch (#237)

## Version 0.2.0 (January 14, 2026)

This release contains a number of breaking changes. See below for details on how
to upgrade from v0.1.19.

### Breaking

 - Remove `::Matrix` method that maps over columns (#223).

   This change means you cannot do `MathOptAI.add_predictor(model, predictor, x)`
   where `x::Matrix` and have it return a `y::Matrix` corresponding to mapping
   the predictor over the columns of `x`.

   To upgrade, replace:
   ```julia
   y, formulation = MathOptAI.add_predictor(model, predictor, x::Matrix)
   ```
   with:
   ```julia
   ret = map(1:size(x, 2)) do i
       return MathOptAI.add_predictor(model, predictor, x[:, i])
   end
   formulation = last.(ret)
   y = reduce(hcat, first.(ret))
   ```
   and write additional code as needed to handle the `formulation` objects.

 - Replace GrayBox by VectorNonlinearOracle implementation (#230)

   We have removed the old `GrayBox` predictor and replaced it by the
   `VectorNonlinearOracle` predictor, which has now been renamed to `GrayBox`.
   In addition, the `; vector_nonlinear_oracle = true` keyword argument has been
   removed, and the new `; gray_box = true` is equivalent to the old
   `; vector_nonlinear_oracle = true`.

   To upgrade, replace:
   ```julia
   MathOptAI.add_predictor(model, predictor, x; vector_nonlinear_oracle = true)
   ```
   with:
   ```julia
   MathOptAI.add_predictor(model, predictor, x; gray_box = true)
   ```

 - Change `config` dictionary to have functions as values (#233)

   We have changed how the `; config = Dict()` keyword works for the Flux, Lux,
   and PyTorch extensions. Where previously the values were instantiated
   [`AbstractPredictor`](@ref) _objects_, they must now be _constructors_ that,
   when called, return an [`AbstractPredictor`](@ref) object.

   As an example of upgrading, replace:
   ```julia
   ; config = Dict(Flux.relu => MathOptAI.ReLU())
   ```
   with:
   ```julia
   ; config = Dict(Flux.relu => MathOptAI.ReLU)
   # or, alternatively
   ; config = Dict(Flux.relu => () -> MathOptAI.ReLU())
   ```

### Added

 - Add [`LeakyReLU`](@ref) predictor (#218)
 - Add [`AvgPool2d`](@ref), [`Conv2d`](@ref), and [`MaxPool2d`](@ref) predictors
   (#220), (#222), (#224)
 - Add support for custom models in Flux (#227)
 - Add [`MaxPool2dBigM`](@ref) predictor (#231), (#235)

### Fixed

 - Simplify test to fix flakey `TestPythonCallExt.test_gelu` (#228)

### Other

 - Remove comment that `vector_nonlinear_oracle` is experimental (#219)
 - Added this changelog (#234), (#236)

## Version 0.1.19 (December 5, 2025)

### Added

 - Add [`replace_weights_with_variables`](@ref) (#215)

## Version 0.1.18 (October 24, 2025)

### Added

 - Propagate start values through layers (#210)

### Other

 - Update to `JuliaFormatter@2` (#211)
 - Make some predictors callable (#212)

## Version 0.1.17 (October 23, 2025)

### Fixed

 - Iterate over PyTorch Sequential module itself, rather than its children (#207)

## Version 0.1.16 (October 22, 2025)

### Added

 - Add [`GELU`](@ref) predictor (#202)
 - Add support for `MOI.VectorNonlinearOracle` (#204)

### Other

 - Clarify the problem class of each predictor (#205)

## Version 0.1.15 (September 30, 2025)

### Added

- Add extension hooks for InfiniteOpt (#200)

### Other

 - Refactor different ReLU into separate files (#199)

## Version 0.1.14 (September 8, 2025)

### Fixed

 - Fix binary decision tree by making the tolerance default to 1e-6 (#196)
 - Fix docs because of a change in Lux (#197)

### Other

 - Update README.md with MOAI paper (#194)

## Version 0.1.13 (July 8, 2025)

### Fixed

 - Add DimensionMismatch checks for Affine and Scale (#192)

## Version 0.1.12 (June 3, 2025)

### Added

 - Add a `reduced_space` fallback for extensions (#187)

### Other

 - Update test for change in MOI@1.40.2 (#186)
 - Add `Softmax` to the PyTorch manual page (#188)

## Version 0.1.11 (June 2, 2025)

### Added

 - Add LinearCombination predictor (#181)
 - Add EvoTrees.jl extension (#182)

### Other

 - Fix docstrings in MathOptAIEvoTreesExt.jl (#183)
 - Update to Lux@1 (#184)

## Version 0.1.10 (April 14, 2025)

### Added

 - Allow relaxation of ReLUQuadratic (#178)

### Fixed

 - Fix duplicate test name (#176)

## Version 0.1.9 (April 1, 2025)

### Added

 - Add `VectorNonlinearOracle` predictor (#172)

## Version 0.1.8 (March 31, 2025)

### Added

 - Set device in `output_size` (#170)
 - Support [`SoftMax`](@ref) in PyTorchModel (#173)

### Other

 - Add paper to README (#169)

## Version 0.1.7 (January 17, 2025)

### Fixed

 - Evaluate torch model to get output dimension (#166)

## Version 0.1.6 (November 7, 2024)

### Other

 - Clarify instructions to load PythonCall (#163)

## Version 0.1.5 (November 6, 2024)

### Added

 - Add `gray_box_device` option for `PyTorchModel` (#159)

### Fixed

 - Fix flakey tests (#161)

## Version 0.1.4 (October 25, 2024)

### Fixed

 - Add bound constraints to the Formulation object (#155)

### Other

 - Improve the docstrings (#156)
 - Fix various typos (#157)

## Version 0.1.3 (October 24, 2024)

### Other

 - Delete `.github/workflows/documentation-deploy.yml` (#150)
 - Improve manual for NN extensions (#152)
 - Fix documentation link in README.md (#153)

## Version 0.1.2 (October 23, 2024)

### Other

 - Add `DOCUMENTER_KEY` (#148)

## Version 0.1.1 (October 23, 2024)

### Other

 - Update installation instructions for registered package (#141)
 - Create documentation-deploy.yml (#146)

## Version 0.1.0 (October 22, 2024)

Initial release. Too many pull requests to reference.
