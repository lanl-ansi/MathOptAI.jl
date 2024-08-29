# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAIPythonCallExt

import JuMP
import PythonCall
import MathOptAI

"""
    MathOptAI.add_predictor(
        model::JuMP.AbstractModel,
        predictor::MathOptAI.PytorchModel,
        x::Vector;
        config::Dict = Dict{Any,Any}(),
        reduced_space::Bool = false,
        gray_box::Bool = false,
    )

Add a trained neural network from Pytorch via PythonCall.jl to `model`.

## Supported layers

 * `nn.Linear`
 * `nn.ReLU`
 * `nn.Sequential`
 * `nn.Sigmoid`
 * `nn.Tanh`

## Keyword arguments

 * `config`: a dictionary that maps `Symbol`s to [`AbstractPredictor`](@ref)s
   that control how the activation functions are reformulated. For example,
   `:Sigmoid => MathOptAI.Sigmoid()` or `:ReLU => MathOptAI.QuadraticReLU()`.
   The supported Symbols are `:ReLU`, `:Sigmoid`, and `:Tanh`.
 * `gray_box`: if `true`, the neural network is added as a user-defined
   nonlinear operator, with gradients provided by `torch.func.jacrev`.
"""
function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
    predictor::MathOptAI.PytorchModel,
    x::Vector;
    config::Dict = Dict{Any,Any}(),
    reduced_space::Bool = false,
    gray_box::Bool = false,
)
    inner_predictor = MathOptAI.build_predictor(predictor; config, gray_box)
    if reduced_space
        inner_predictor = MathOptAI.ReducedSpace(inner_predictor)
    end
    return MathOptAI.add_predictor(model, inner_predictor, x)
end

"""
    MathOptAI.build_predictor(
        predictor::MathOptAI.PytorchModel;
        config::Dict = Dict{Any,Any}(),
        gray_box::Bool = false,
    )

Convert a trained neural network from Pytorch via PythonCall.jl to a
[`Pipeline`](@ref).

## Supported layers

 * `nn.Linear`
 * `nn.ReLU`
 * `nn.Sequential`
 * `nn.Sigmoid`
 * `nn.Tanh`

## Keyword arguments

 * `config`: a dictionary that maps `Symbol`s to [`AbstractPredictor`](@ref)s
   that control how the activation functions are reformulated. For example,
   `:Sigmoid => MathOptAI.Sigmoid()` or `:ReLU => MathOptAI.QuadraticReLU()`.
   The supported Symbols are `:ReLU`, `:Sigmoid`, and `:Tanh`.
 * `gray_box`: if `true`, the neural network is added as a user-defined
   nonlinear operator, with gradients provided by `torch.func.jacrev`.
"""
function MathOptAI.build_predictor(
    predictor::MathOptAI.PytorchModel;
    config::Dict = Dict{Any,Any}(),
    gray_box::Bool = false,
)
    if gray_box
        if !isempty(config)
            error("cannot specify the `config` kwarg if `gray_box = true`")
        end
        return MathOptAI.GrayBox(predictor)
    end
    torch = PythonCall.pyimport("torch")
    nn = PythonCall.pyimport("torch.nn")
    torch_model = torch.load(predictor.filename)
    return _predictor(nn, torch_model, config)
end

function _predictor(nn, layer, config)
    if Bool(PythonCall.pybuiltins.isinstance(layer, nn.Linear))
        weight = mapreduce(vcat, layer.weight.tolist()) do w
            return PythonCall.pyconvert(Vector{Float64}, w)'
        end
        bias = PythonCall.pyconvert(Vector{Float64}, layer.bias.tolist())
        return MathOptAI.Affine(Matrix(weight), bias)
    elseif Bool(PythonCall.pybuiltins.isinstance(layer, nn.ReLU))
        return get(config, :ReLU, MathOptAI.ReLU())
    elseif Bool(PythonCall.pybuiltins.isinstance(layer, nn.Sequential))
        layers = [_predictor(nn, child, config) for child in layer.children()]
        return MathOptAI.Pipeline(layers)
    elseif Bool(PythonCall.pybuiltins.isinstance(layer, nn.Sigmoid))
        return get(config, :Sigmoid, MathOptAI.Sigmoid())
    elseif Bool(PythonCall.pybuiltins.isinstance(layer, nn.Tanh))
        return get(config, :Tanh, MathOptAI.Tanh())
    end
    return error("unsupported layer: $layer")
end

function MathOptAI.GrayBox(predictor::MathOptAI.PytorchModel)
    torch = PythonCall.pyimport("torch")
    torch_model = torch.load(predictor.filename)
    J = torch.func.jacrev(torch_model)
    # TODO(odow): I'm not sure if there is a better way to get the output
    # dimension of a torch model object?
    output_size(::Any) = PythonCall.pyconvert(Int, torch_model[-1].out_features)
    function with_jacobian(x)
        py_x = torch.tensor(collect(x))
        py_value = torch_model(py_x).detach().numpy()
        py_jacobian = J(py_x).detach().numpy()
        return (;
            value = PythonCall.pyconvert(Vector, py_value),
            jacobian = PythonCall.pyconvert(Matrix, py_jacobian),
        )
    end
    return MathOptAI.GrayBox(output_size, with_jacobian)
end

end  # module
