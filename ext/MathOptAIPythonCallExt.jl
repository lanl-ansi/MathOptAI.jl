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
        model::JuMP.Model,
        predictor::MathOptAI.PytorchModel,
        x::Vector;
        config::Dict = Dict{Any,Any}(),
        reduced_space::Bool = false,
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
"""
function MathOptAI.add_predictor(
    model::JuMP.Model,
    predictor::MathOptAI.PytorchModel,
    x::Vector;
    config::Dict = Dict{Any,Any}(),
    reduced_space::Bool = false,
)
    inner_predictor = MathOptAI.build_predictor(predictor; config)
    if reduced_space
        # If config maps to a ReducedSpace predictor, we'll get a MethodError
        # when trying to add the nested redcued space predictors.
        # TODO: raise a nicer error or try to handle this gracefully.
        inner_predictor = MathOptAI.ReducedSpace(inner_predictor)
    end
    return MathOptAI.add_predictor(model, inner_predictor, x)
end

"""
    MathOptAI.build_predictor(
        predictor::MathOptAI.PytorchModel;
        config::Dict = Dict{Any,Any}(),
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
"""
function MathOptAI.build_predictor(
    predictor::MathOptAI.PytorchModel;
    config::Dict = Dict{Any,Any}(),
)
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
        return MathOptAI.Affine(weight, bias)
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

end  # module
