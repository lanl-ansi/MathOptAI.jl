# Copyright (c) 2024: Oscar Dowson and contributors                         #src
# Copyright (c) 2024: Triad National Security, LLC                          #src
#                                                                           #src
# Use of this source code is governed by a BSD-style license that can be    #src
# found in the LICENSE.md file.                                             #src

# # Input Convex Neural Networks with PyTorch

# The purpose of this tutorial is to explain how to embed an input convex
# neural network model from [PyTorch](https://pytorch.org) into JuMP.

# !!! info
#     To use PyTorch from MathOptAI, you must first follow the
#     [Python integration](@ref) instructions.

# ## Required packages

# This tutorial requires the following packages

using JuMP
using HiGHS
using MathOptAI
using PythonCall
import Plots
import Random

# ## Building the ICNN

# The following custom layer can be used to build ICNNs. This layer has two
# forward methods. One that takes a single input and the other takes  a `Tuple`.
# They both return the result of the forward pass as well as the original input.

dir = mktempdir(".")

write(
    joinpath(dir, "icnn.py"),
    """
    import math
    import torch
    from torch.nn.parameter import Parameter
    from torch.nn import functional as F, init

    class InputConvex(torch.nn.Module):
        def __init__(
            self,
            in_features_z: int,
            in_features_x: int,
            out_features: int,
            bias: bool = True,
            activation = F.relu, 
            device=None,
            dtype=None,
        ):
            factory_kwargs = {"device": device, "dtype": dtype}
            super().__init__()
            self.activation = activation
            self.in_features_z = in_features_z
            self.in_features_x = in_features_x
            self.out_features = out_features
            self.weight_z = Parameter(
                torch.empty((out_features, in_features_z), **factory_kwargs)
            )
            self.weight_x = Parameter(
                torch.empty((out_features, in_features_x), **factory_kwargs)
            )
            if bias:
                self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            init.kaiming_uniform_(self.weight_z, a=math.sqrt(5))
            init.kaiming_uniform_(self.weight_x, a=math.sqrt(5))
            if self.bias is not None:
                fan_in_z, _ = init._calculate_fan_in_and_fan_out(self.weight_z)
                fan_in_x, _ = init._calculate_fan_in_and_fan_out(self.weight_x)
                bound_z = 1 / math.sqrt(fan_in_z) if fan_in_z > 0 else 0
                bound_x = 1 / math.sqrt(fan_in_x) if fan_in_x > 0 else 0
                init.uniform_(self.bias, -bound_z, bound_z)
                init.uniform_(self.bias, -bound_x, bound_x)
                
        def forward(self, *args):
            if len(args) == 1 and isinstance(args[0], tuple):
                args = args[0]
            if len(args) == 1:
                input_x = args[0]
                output = self.activation(input_x @ self.weight_x.T + self.bias)
                return output, input_x
            elif len(args) == 2:
                input_z, input_x = args
                output = self.activation(
                    input_z @ F.softplus(self.weight_z).T + 
                    input_x @ self.weight_x.T + 
                    self.bias
                )
                return output, input_x

    class InputConvexChain(torch.nn.Module):
        def __init__(self, *layers):
            super(InputConvexChain, self).__init__()
            self.layers = torch.nn.ModuleList(layers)
        def forward(self, x):
            layer1 = self.layers[0]
            z, x = layer1(x)
            for layer in self.layers[1:]:
                if isinstance(layer, InputConvex):
                    z, x = layer(z, x)
                else:
                    z = layer(z)
            return z
    """,
)

filename = joinpath(dir, "icnn.pt")

# Next, we import the network and the layers using `PythonCall.@pyexec`:

predictor, InputConvex, InputConvexChain = PythonCall.@pyexec(
    (dir, filename) =>
        """
        import torch
        from torch.nn import ReLU
        import sys
        sys.path.insert(0, dir)
        from icnn import InputConvexChain, InputConvex
        predictor = InputConvexChain(
            InputConvex(32, 32, 8), 
            ReLU(), 
            InputConvex( 8, 32, 1), 
            ReLU(), 
        )
        torch.save(predictor, filename)
        """ => (predictor, InputConvex, InputConvexChain)
)

# Let's test the ICNN
torch = PythonCall.pyimport("torch")
predictor(torch.rand(32))

# ## Building the Predictor

# To provide a description for embedding `InputConvexChain`
# into JuMP, we create the following callback function:

function icnn_callback(icnn::PythonCall.Py; input_size, kwargs...)
    p = Pipeline(AbstractPredictor[])
    softplus = SoftPlus()
    nn = PythonCall.pyimport("torch.nn")
    for (i, layer) in enumerate(icnn.layers)
        if i == 1
            w_x =
                pyconvert(Array{Float64}, layer.weight_x.detach().cpu().numpy())
            b = pyconvert(Array{Float64}, layer.bias.detach().cpu().numpy())
            push!(p.layers, Affine(w_x, b))
        else
            if pyisinstance(layer, InputConvex)
                w_x = pyconvert(
                    Array{Float64},
                    layer.weight_x.detach().cpu().numpy(),
                )
                w_z = pyconvert(
                    Array{Float64},
                    layer.weight_z.detach().cpu().numpy(),
                )
                b = pyconvert(Array{Float64}, layer.bias.detach().cpu().numpy())
                push!(p.layers, Affine([softplus.(w_z) w_x], b))
            else
                append!(
                    p.layers,
                    MathOptAI.build_predictor(nn.Sequential(layer); kwargs...).layers,
                )
            end
        end
    end
    return InputConvexChainPredictor(p)
end

# In addition, we need to implement and [`add_predictor`](@ref) for
# `InputConvexChain` in order to be able to embed this network into JuMP.
# For this purpose, we define `InputConvexChainPredictor` and implement
# [`add_predictor`](@ref):

struct InputConvexChainPredictor <: MathOptAI.AbstractPredictor
    p::Pipeline
end

function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
    predictor::InputConvexChainPredictor,
    x::Vector;
    kwargs...,
)::Tuple{<:Vector,<:AbstractFormulation}
    layer1 = first(predictor.p.layers)
    formulation = PipelineFormulation(predictor, [])
    z, inner_formulation = MathOptAI.add_predictor(model, layer1, x; kwargs...)
    push!(formulation.layers, inner_formulation)
    for layer in predictor.p.layers[2:end]
        if layer isa Affine
            z, inner_formulation =
                MathOptAI.add_predictor(model, layer, [z; x]; kwargs...)
        else
            z, inner_formulation =
                MathOptAI.add_predictor(model, layer, z; kwargs...)
        end
        push!(formulation.layers, inner_formulation)
    end
    return z, formulation
end

# ## Embed ICNN into JuMP

model = Model()
@variable(model, x[1:32])

#-

config = Dict(:ReLU => ReLUSOS1, InputConvexChain => icnn_callback)
z, formulation = MathOptAI.add_predictor(model, predictor, x; config)

#-

z

#-

formulation

# ## Epigraph formulations

# The nice thing about ICNNs is that we can formulate their epigraph and avoid
# adding binary variables to the model. For that, we can use
# [`ReLUEpigraph`](@ref).

# Let's create a PyTorch model with scalar input:

predictor = PythonCall.@pyexec(
    (dir, filename) =>
        """
        import torch
        from torch.nn import ReLU
        import sys
        sys.path.insert(0, dir)
        from icnn import InputConvexChain, InputConvex
        torch.manual_seed(61)
        predictor = InputConvexChain(
            InputConvex(1, 1, 8), 
            ReLU(), 
            InputConvex(8, 1, 1), 
            ReLU(), 
        )

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(predictor.parameters(), lr=0.01, momentum=.9)
        predictor.train()
        X = torch.unsqueeze(torch.arange(-2, 2, step=.1), 1)
        Y = torch.pow(X, 2)
        epochs = 100
        running_loss = 0.
        for e in range(epochs):
            optimizer.zero_grad()
            Y_hat = predictor(X)
            loss = loss_fn(Y_hat, Y)
            loss.backward()
            optimizer.step()
            if e % 10 == 9:
                last_loss = running_loss # loss per batch
                print(f'  batch {e + 1} loss: {loss.item()}')

        torch.save(predictor, filename)
        """ => predictor
)

# Next, we use [`ReLUEpigraph`](@ref) to embed this ICNN into JuMP.

model = Model(HiGHS.Optimizer)
set_silent(model)
@variable(model, x[1:1])
config = Dict(:ReLU => ReLUEpigraph, InputConvexChain => icnn_callback)
y, _ = MathOptAI.add_predictor(model, predictor, x; config)
@objective(model, Min, only(y))
model

# Because we used the [`ReLUEpigraph`](@ref) predictor, there are no binary or
# integer variables in our model.
#
# Moreover, we can show that the objective value `y` is convex with respect to
# `x`:

x_value, y_value = -2:0.1:2, Float64[]
for xi in x_value
    fix(x[1], xi)
    optimize!(model)
    ## To prove we are solving an LP and not a MIP, require dual solutions.
    assert_is_solved_and_feasible(model; dual = true)
    push!(y_value, objective_value(model))
end
Plots.plot(x_value, y_value; xlabel = "x", ylabel = "y", label = "Trained")
Plots.plot!(x_value, x_value .^ 2; label = "Target", linestyle = :dash)
