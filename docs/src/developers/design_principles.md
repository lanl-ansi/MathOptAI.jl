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

### MathOptAI

The main entry-point to MathOptAI is [`add_predictor`](@ref):
```julia
y = MathOptAI.add_predictor(model, predictor, x)
```
The user provides the input `x`, and the output `y` is returned.

The main benefit of this approach is simplicity.

First, the user probably already has the input `x` as decision variables or an
expression in the model, so we do not need the `connect_input` constraint, and
because we use a full-space formulation, the output `y` will always be a vector
of decision variables, which avoids the need for a `connect_output` constraint.

Second, predictors do not need to store dimension information, so we can have:
```julia
y = MathOptAI.add_predictor(model, MathOptAI.ReLU(), x)
```
for any size of `x`.

The main downsides are that we do not return a `pred_constr` equivalent
contains statistics on the reformulation, and that the user cannot delete a
predictor from a model once added.

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
y = MathOptAI.add_predictor(model, MathOptAI.ReLU(), x)
```
This means that we can pipeline them to create predictors such as:
```julia
function LogisticRegression(A)
    return MathOptAI.Pipeline(MathOptAI.Affine(A), MathOptAI.Sigmoid())
end
```
