# Checklists

The purpose of this page is to collate a series of checklists for commonly
performed changes to the source code of MathOptAI.

In each case, copy the checklist into the description of the pull request.

## Making a release

In preparation for a release, use the following checklist. These steps can be
done in the same commit, or separately. The last commit should have the message
"Prep for vX.Y.Z."

````
## Pre-release

 - [ ] Update `docs/src/changelog.md`
 - [ ] Change the version number in `Project.toml`
 - [ ] The commit messages in this PR do not contain `[ci skip]`

## The release

 - [ ] After merging this pull request, comment `[at]JuliaRegistrator register`
       in the GitHub commit.
````

## Adding a new predictor

Use this checklist when adding a new predictor.

````
## The predictor

 - [ ] Create a new file in `src/predictors`
 - [ ] Add the default copyright header
 - [ ] Make a new `<: AbstractPredictor`  type
 - [ ] Add a docstring to the predictor
 - [ ] Add the predictor to `docs/src/api.md`
 - [ ] Add the predictor to `docs/src/manual/predictors.md`
 - [ ] Implement `output_size`
 - [ ] Implement `add_predictor(model, ::NewPredictor, x)`
 - [ ] Optionally implement `add_predictor(model, ::ReducedSpace{NewPredictor}, x)`
 - [ ] Add a test to `test/test_predictors.jl`

## Extensions

For each revelevant extension:

 - [ ] Add support for the new predictor
 - [ ] Add tests
 - [ ] Mention the predictor in the docstring of the relevant `build_predictor`
 - [ ] Mention the predictor in the relevant `docs/src/manual/<ext>.md`
````
