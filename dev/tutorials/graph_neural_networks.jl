# Copyright (c) 2024: Oscar Dowson and contributors                         #src
# Copyright (c) 2024: Triad National Security, LLC                          #src
#                                                                           #src
# Use of this source code is governed by a BSD-style license that can be    #src
# found in the LICENSE.md file.                                             #src

if get(ENV, "LOGNAME", "") == "odow"                                        #src
    ENV["JULIA_PYTHONCALL_EXE"] = "python3"                                 #src
    ENV["JULIA_CONDAPKG_BACKEND"] = "Null"                                  #src
end                                                                         #src

# # Graph neural networks

# The purpose of this tutorial is to explain how to embed graph neural network
# models from [PyTorch Geometric](https://pytorch-geometric.readthedocs.io) into
# JuMP.

# !!! info
#     To use PyTorch from MathOptAI, you must first follow the
#     [Python integration](@ref) instructions.

# ## Required packages

# This tutorial requires the following packages

using JuMP
using Test
import Ipopt
import MathOptAI
import PythonCall

# ## Setting up the Python side

# We have written a custom PyTorch module, which is stored in the file
# `my_gnn.py`:

print(read(joinpath(@__DIR__, "my_gnn.py"), String))

# To make it easy to load this Python file into the documentation, we add the
# current directory to the Python path:

dir = @__DIR__
PythonCall.@pyexec(dir => "import sys; sys.path.insert(0, dir)")

# Then, we can import torch and the `my_gnn` file into Julia:

my_gnn = PythonCall.pyimport("my_gnn");

# Now, we can load our GNN into Julia:

predictor = my_gnn.MyGNN()

# In practice, you should train the GNN in Python, and write it to a file using
# `torch.save`; see the [Function fitting with PyTorch](@ref) for details. You
# could then replace the `predictor` with an appropriate [`PytorchModel`](@ref).

# ## Creating the graph

# Before we can embed `predictor` into a JuMP model, we need to specialize it to
# a particular graph structure, and we need to provide a callback function that
# MathOptAI can use to turn an instance of `MyGNN` into an
# [`AbstractPredictor`](@ref).

# First, we need the graph structure. Rather than providing a 2-by-n tensor, we
# provide the edges as a list of `i => j` pairs:

edge_index = [1 => 2, 2 => 1, 2 => 3, 3 => 2, 3 => 4, 4 => 3]

# Note that the nodes are 1-indexed.

# The callback takes in a `PythonCall.Py` layer, and returns an
# [`AbstractPredictor`](@ref) that matches the forward pass of the GNN:

function MyGNN_callback(layer::PythonCall.Py; kwargs...)
    return MathOptAI.Pipeline(
        MathOptAI.GCNConv(layer.conv; edge_index),
        MathOptAI.Sigmoid(),
    )
end

# Note how the callback uses `edge_index`.

# ## Creating the JuMP model

# Now we can embed `predictor` into a JuMP model.

model = Model(Ipopt.Optimizer)
set_silent(model)

# Our input `x` has four rows, one for each node, and two columns, one for each
# input attribute. The fixed bounds are so we can easily compare solutions
# between Python and Julia; replace them with your real constraints in practice.

@variable(model, x[i in 1:4, j in 1:2] == sin(i) + cos(j))

# Then, we add `predictor` with the `config` argument mapping `gnn.MyGNN` (the
# Python class) to our callback `MyGNN_callback`:

y, _ = MathOptAI.add_predictor(
    model,
    predictor,
    x;
    config = Dict(my_gnn.MyGNN => MyGNN_callback),
);

# Now we can solve the model:

optimize!(model)
assert_is_solved_and_feasible(model)

# and look at the solution of `y`:

Y = reshape(value(y), 4, 3)

# Which is identical to what we obtain when we evaluate the GNN in Python:

torch = PythonCall.pyimport("torch")
py_x = torch.tensor([value.(x[i, :]) for i in 1:4])
## `py_edge_index` is the same graph as `edge_index`, just in a different form.
## Note also that this is 0-indexed.
py_edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
predictor(py_x, py_edge_index)
