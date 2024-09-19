```@meta
CurrentModule = MathOptAI
```

# Design principles

This project is inspired by two existing projects:

 * [OMLT](https://github.com/cog-imperial/OMLT)
 * [gurobi-machinelearning](https://github.com/Gurobi/gurobi-machinelearning)

OMLT is a framework built around [Pyomo](https://pyomo.org), and
gurobi-machinelearning is a framework build around gurobipy.

These projects served as inspiration, but we also departed from them in some
carefully considered ways.

All of our design decisions were guided by two principles:

 1. To be simple
 2. To leverage Julia's `Pkg` extensions and multiple dispatch.

## Terminology

Because the field is relatively new, there is no settled choice of terminology.

MathOptAI chooses to use "predictor" as the synonym for the machine learning
model. Hence, we have `AbstractPredictor`, `add_predictor`, and
`build_predictor`.

In contrast, gurob-machinelearning tennds to use "regression model" and OMLT
does not have a single unified API.

We choose "predictor" because all models we implement are of the form
``y = f(x)``.

We do not use "machine learning model" because we have support for the linear
and logistic regression models of classical statistical fitting. We could have
used "regression model", but we find that models like neural networks and
binary decision trees are not commonly thought of as regression models.

## Inputs are vectors

MathOptAI assumes that all inputs ``x`` and outputs ``y`` to ``y = predictor(x)``
are `Base.Vector`s.

We make this choice for simplicity.

In our opinion, Julia libraries often take a laissez-faire approach to the types
that they support. In the optimistic case, this can lead to novel behavior by
combining two packages that the package author had previously not considered or
tested. In the pessimistic case, this can lead to incorrect results or cryptic
error messagges.

Exceptions to the `Vector` rule will be carefully considered and tested.

Currently, there are two exceptions:

 1. If `x` is a `Matrix`, then the columns of `x` are interpreted as independent
    observations, and the output `y` will be a `Matrix` with the same number of
    columns
 2. The `StatsModels` extension allows `x` to be a `DataFrames.DataFrame`, if
    the predictor is a `StatsModels.TableRegressionModel`.

Exceptions 1 and 2 are combined in the `StatsModels` exception, so that the
predictor is mapped over the rows of the `DataFrames.DataFrame` (which we assume
will be a common use-case).

We choose to interpret the rows as input variables and columns as independent
observations (rather than the more traditional table-based approach where
columns are the input variables and rows are observations) because Julia uses
column-major ordering in `Matrix`. Another justification follows from the
[`Affine`](@ref) predictor, ``f(x) = Ax + b``, where passing in a `Matrix` as
`x` with column observations naturally leads to a `Matrix` output for `y` of the
appropriate dimensions.

We choose to make `y` a `Vector`, even for scalar outputs, to simplify code that
works generically for many different predictors. Without this principle, there
will inevitably be cases where a scalar and length-1 vector are confused.

If you want to use a predictor that does not take `Vector` input (for example,
it is an image as input to a neural network), the first preprocessing step
should be to `vec` the input into a single `Vector`.

## Inputs are provided, outputs are returned

The main job of MathOptAI is to embed models of the form `y = predictor(x)` into
a JuMP model. A key design decision is how to represent the input `x` and output
`y`.

### gurobi-machinelearning

gurobi-machinelearning implements an API of the following general form:
```python
pred_constr = add_predictor_constr(model, predictor, x, y)
```
Here, both the input `x` and the output `y` must be created and provided by the
user, and a new object `pred_constr` is returned.

The benefit of this design is that `pred_constr` can contain statistics about
the reformulation (for example, the number of variables that were added), and it
can be used to delete a predictor from `model`.

The downside is that the user must ensure that the shape and size of `y` is
correct.

### OMLT

OMLT implements an API of the following general form:
```python
model.pred_constr = OmltBlock()
model.pred_constr.build_formulation(predictor)
x, y = model.pred_constr.inputs, model.pred_constr.outputs
```
First, a new `OmltBlock()` is created. Then the formulation is built inside the
block, and both the input and output are provided to the user.

The benefit of this design is that `pred_constr` can contain statistics about
the reformulation (for example, the number of variables that were added), and it
can be used to delete a predictor from `model`.

The downside is that the user must often write additional constraints to connect
the input and output of the `OmltBlock` to their existing decision variables:

```python
#connect pyomo model input and output to the neural network
@model.Constraint()
def connect_input(mdl):
    return mdl.input == mdl.nn.inputs[0]

@model.Constraint()
def connect_output(mdl):
    return mdl.output == mdl.nn.outputs[0]
```

A second downside is that the predictor must describe the input and output
dimension; these cannot be inferred automatically. As one example, this means
that it cannot do the following:
```python
# Not possible because dimension not given
model.pred_constr.build_formulation(ReLU())
```
In the context of MathOptAI, something like `ReLU()` is useful so that we can
map generic layers like `Flux.relu => MathOptAI.ReLU()`, and so that we do not
duplicate required dimension information in input and predictor (see the
MathOptAI section below).

### MathOptAI

The main entry-point to MathOptAI is [`add_predictor`](@ref):
```julia
y, formulation = MathOptAI.add_predictor(model, predictor, x)
```
The user provides the input `x`, and the output `y` is returned.

The main benefit of this approach is simplicity.

First, the user probably already has the input `x` as decision variables or an
expression in the model, so we do not need the `connect_input` constraint, and
because we use a full-space formulation by default, the output `y` will always
be a vector of decision variables, which avoids the need for a `connect_output`
constraint.

Second, predictors do not need to store dimension information, so we can have:
```julia
y, formulation = MathOptAI.add_predictor(model, MathOptAI.ReLU(), x)
```
for any size of `x`.

We choose this decision to simplify the implementation, and because we think
deleting a predictor is an uncommon operation.

## Activations are predictors

OMLT makes a distinction between layers, like `full_space_dense_layer`, and
elementwise activation functions, like `sigmoid_activation_function`.

The downside to this approach is that it treats activation functions as special,
leading to issues such as [OMLT#125](https://github.com/cog-imperial/OMLT/issues/125).

In constrast, MathOptAI treats activation functions as a vector-valued predictor
like any other:
```julia
y, formulation = MathOptAI.add_predictor(model, MathOptAI.ReLU(), x)
```
This means that we can pipeline them to create predictors such as:
```julia
function LogisticRegression(A)
    return MathOptAI.Pipeline(MathOptAI.Affine(A), MathOptAI.Sigmoid())
end
```

## Controlling transformations

Many predictors have multiple ways that they can be formulated in an
optimization model. For example, [`ReLU`](@ref) implements the non-smooth
nonlinear formulation ``y = \max\{x, 0\}``, while [`ReLUQuadratic`](@ref)
implements a the complementarity formulation
``x = y - slack; y, slack \\ge 0; y * slack == 0``.

Choosing the appropriate formulation for the combination of model and solver can
have a large impact on the performance.

Because gurobi-machinelearning is specific to the Gurobi solver, they have a
limited ability for the user to choose and implement different formulations.

OMLT is more general, in that it has multiple ways of formulating layers such
as ReLU. However, these are hard-coded into complete formulations such as
`omlt.neuralnet.nn_formulation.ReluBigMFormulation` or
`omlt.neuralnet.nn_formulation.ReluComplementarityFormulation`.

In contrast, MathOptAI tries to take a maximally modular approach, where the
user can control how the layers are formulated at runtime, including using a
custom formulation that is not defined in MathOptAI.jl.

Currently, we achive this with a `config` dictionary, which maps the various
neural network layers to an [`AbstractPredictor`](@ref). For example:
```julia
chain = Flux.Chain(Flux.Dense(1 => 16, Flux.relu), Flux.Dense(16 => 1));
config = Dict(Flux.relu => MathOptAI.ReLU())
predictor = MathOptAI.build_predictor(chain; config)
```
Please open a GitHub issue if you have a suggestion for a better API.

## Full-space or reduced-space

OMLT has two ways that it can formulate neural networks: full-space and
reduced-space.

The full-space formulations add intermediate variables to represent the output
of all layers.

For example, in a `Flux.Dense(2, 3, Flux.relu)` layer, a full-space formulation
will add an intermediate `y_tmp` variable to represent the output of the affine
layer prior to the ReLU:
```julia
layer = Flux.Dense(2, 3, Flux.relu)
model_full_space = Model()
@variable(model_full_space, x[1:2])
@variable(model_full_space, y_tmp[1:3])
@variable(model_full_space, y[1:3])
@constraint(model_full_space, y_tmp == layer.A * x + layer.b)
@constraint(model_full_space, y .== max.(0, y_tmp))
```

In contrast, a reduced-space formulation encodes the input-output relationship
as a single nonlinear constraint:
```julia
layer = Flux.Dense(2, 3, Flux.relu)
model_reduced_space = Model()
@variable(model_reduced_space, x[1:2])
@variable(model_reduced_space, y[1:3])
@constraint(model_reduced_space, y .== max.(0, layer.A * x + layer.b))
```

In general, the full-space formulations have more variables and constraints but
simpler nonlinear expressions, whereas the reduced-space formulations have fewer
variables and constraints but more complicated nonlinear expressions.

MathOptAI.jl implements the full-space formulation by default, but some layers
support the reduced-space formulation with the [`ReducedSpace`](@ref) wrapper.
