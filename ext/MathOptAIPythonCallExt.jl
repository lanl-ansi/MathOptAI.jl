# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
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
        vector_nonlinear_oracle::Bool = false,
        hessian::Bool = vector_nonlinear_oracle,
        device::String = "cpu",
    )

Add a trained neural network from PyTorch via PythonCall.jl to `model`.

## Supported layers

 * `nn.Linear`
 * `nn.ReLU`
 * `nn.Sequential`
 * `nn.Sigmoid`
 * `nn.Softmax`
 * `nn.Softplus`
 * `nn.Tanh`

## Keyword arguments

 * `config`: a dictionary that maps `Symbol`s to [`AbstractPredictor`](@ref)s
   that control how the activation functions are reformulated. For example,
   `:Sigmoid => MathOptAI.Sigmoid()` or `:ReLU => MathOptAI.QuadraticReLU()`.
   The supported Symbols are `:ReLU`, `:Sigmoid`, `:SoftMax`, `:SoftPlus`, and
   `:Tanh`.

 * `reduced_space`: if `true`, the neural network is added using a
   [`ReducedSpace`](@ref) formulation.

 * `gray_box`: if `true`, the neural network is added using a [`GrayBox`](@ref)
   formulation.

 * `vector_nonlinear_oracle`: if `true`, the neural network is added using
   `Ipopt._VectorNonlinearOracle`. This is an experimental feature that may
   offer better performance than `gray_box`.

 * `hessian`: if `true`, the `gray_box` and `vector_nonlinear_oracle`
   formulations compute the Hessian of the output using `torch.func.hessian`.
   The default for `hessian` is `false` if `gray_box` is used, and `true` if
   `vector_nonlinear_oracle` is used.

 * `device`: device used to construct PyTorch tensors, for example, `"cuda"`
   to run on an Nvidia GPU.

## Compatibility

The `vector_nonlinear_oracle` feature is experimental. It relies on a private
API feature of Ipopt.jl that will change in a future release.

If you use this feature, you must pin the version of Ipopt.jl in your
`Project.toml` to ensure that future updates to Ipopt.jl do not break your
existing code.

A known good version of Ipopt.jl is v1.8.0. Pin the version using:
```
[compat]
Ipopt = "=1.8.0"
```
"""
function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
    predictor::MathOptAI.PytorchModel,
    x::Vector;
    config::Dict = Dict{Any,Any}(),
    reduced_space::Bool = false,
    kwargs...,
)
    inner_predictor = MathOptAI.build_predictor(predictor; config, kwargs...)
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
        vector_nonlinear_oracle::Bool = false,
        hessian::Bool = vector_nonlinear_oracle,
        device::String = "cpu",
    )

Convert a trained neural network from PyTorch via PythonCall.jl to a
[`Pipeline`](@ref).

## Supported layers

 * `nn.Linear`
 * `nn.ReLU`
 * `nn.Sequential`
 * `nn.Sigmoid`
 * `nn.Softmax`
 * `nn.Softplus`
 * `nn.Tanh`

## Keyword arguments

 * `config`: a dictionary that maps `Symbol`s to [`AbstractPredictor`](@ref)s
   that control how the activation functions are reformulated. For example,
   `:Sigmoid => MathOptAI.Sigmoid()` or `:ReLU => MathOptAI.QuadraticReLU()`.
   The supported Symbols are `:ReLU`, `:Sigmoid`, `:SoftMax`, `:SoftPlus`, and
   `:Tanh`.

 * `gray_box`: if `true`, the neural network is added using a [`GrayBox`](@ref)
   formulation.

 * `vector_nonlinear_oracle`: if `true`, the neural network is added using
   `Ipopt._VectorNonlinearOracle`. This is an experimental feature that may
   offer better performance than `gray_box`.

 * `hessian`: if `true`, the `gray_box` and `vector_nonlinear_oracle`
   formulations compute the Hessian of the output using `torch.func.hessian`.
   The default for `hessian` is `false` if `gray_box` is used, and `true` if
   `vector_nonlinear_oracle` is used.

 * `device`: device used to construct PyTorch tensors, for example, `"cuda"`
   to run on an Nvidia GPU.
"""
function MathOptAI.build_predictor(
    predictor::MathOptAI.PytorchModel;
    config::Dict = Dict{Any,Any}(),
    gray_box::Bool = false,
    vector_nonlinear_oracle::Bool = false,
    device::String = "cpu",
    hessian::Bool = vector_nonlinear_oracle,
    # For backwards compatibility
    gray_box_hessian::Bool = false,
    gray_box_device::Union{Nothing,String} = nothing,
)
    if vector_nonlinear_oracle
        if gray_box
            error(
                "cannot specify `gray_box = true` if `vector_nonlinear_oracle = true`",
            )
        elseif !isempty(config)
            error(
                "cannot specify the `config` kwarg if `vector_nonlinear_oracle = true`",
            )
        end
        return MathOptAI.VectorNonlinearOracle(
            predictor;
            hessian = hessian | gray_box_hessian,
            device = something(gray_box_device, device),
        )
    end
    if gray_box
        if !isempty(config)
            error("cannot specify the `config` kwarg if `gray_box = true`")
        end
        return MathOptAI.GrayBox(
            predictor;
            hessian = hessian | gray_box_hessian,
            device = something(gray_box_device, device),
        )
    end
    torch = PythonCall.pyimport("torch")
    nn = PythonCall.pyimport("torch.nn")
    torch_model = torch.load(predictor.filename; weights_only = false)
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
    elseif Bool(PythonCall.pybuiltins.isinstance(layer, nn.Softmax))
        return get(config, :SoftMax, MathOptAI.SoftMax())
    elseif Bool(PythonCall.pybuiltins.isinstance(layer, nn.Softplus))
        beta = PythonCall.pyconvert(Float64, layer.beta)
        return get(config, :SoftPlus, MathOptAI.SoftPlus(; beta = beta))
    elseif Bool(PythonCall.pybuiltins.isinstance(layer, nn.Tanh))
        return get(config, :Tanh, MathOptAI.Tanh())
    end
    return error("unsupported layer: $layer")
end

function MathOptAI.GrayBox(
    predictor::MathOptAI.PytorchModel;
    hessian::Bool = false,
    device::String = "cpu",
)
    torch = PythonCall.pyimport("torch")
    torch_model = torch.load(predictor.filename; weights_only = false)
    torch_model = torch_model.to(device)
    J = torch.func.jacrev(torch_model)
    H = torch.func.hessian(torch_model)
    function output_size(x::Vector)
        # Get the output size by passing a zero vector through the torch model.
        # We do this instead of `torch_model[-1].out_features` as the last layer
        # may not support out_features.
        z = torch.zeros(length(x); device = device)
        y = torch_model(z)
        return PythonCall.pyconvert(Int, PythonCall.pybuiltins.len(y))
    end
    function callback(x)
        py_x = torch.tensor(collect(x); device = device)
        py_value = torch_model(py_x).detach().cpu().numpy()
        value = PythonCall.pyconvert(Vector, py_value)
        py_jacobian = J(py_x).detach().cpu().numpy()
        jacobian = PythonCall.pyconvert(Matrix, py_jacobian)
        if !hessian
            return (; value, jacobian)
        end
        hessians = PythonCall.pyconvert(Array, H(py_x).detach().cpu().numpy())
        return (; value, jacobian, hessian = permutedims(hessians, (2, 3, 1)))
    end
    return MathOptAI.GrayBox(output_size, callback; has_hessian = hessian)
end

end  # module
