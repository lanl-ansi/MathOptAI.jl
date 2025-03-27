# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAIPythonCallExt

import JuMP
import Ipopt
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
        gray_box_hessian::Bool = false,
        gray_box_device::String = "cpu",
    )

Add a trained neural network from PyTorch via PythonCall.jl to `model`.

## Supported layers

 * `nn.Linear`
 * `nn.ReLU`
 * `nn.Sequential`
 * `nn.Sigmoid`
 * `nn.Softplus`
 * `nn.Tanh`

## Keyword arguments

 * `config`: a dictionary that maps `Symbol`s to [`AbstractPredictor`](@ref)s
   that control how the activation functions are reformulated. For example,
   `:Sigmoid => MathOptAI.Sigmoid()` or `:ReLU => MathOptAI.QuadraticReLU()`.
   The supported Symbols are `:ReLU`, `:Sigmoid`, `:SoftPlus`, and `:Tanh`.
 * `gray_box`: if `true`, the neural network is added as a user-defined
   nonlinear operator, with gradients provided by `torch.func.jacrev`.
 * `gray_box_hessian`: if `true`, the gray box additionally computes the Hessian
   of the output using `torch.func.hessian`.
 * `gray_box_device`: device used to construct PyTorch tensors, e.g. `"cuda"`
   to run on an Nvidia GPU.
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
        gray_box_hessian::Bool = false,
        gray_box_device::String = "cpu",
    )

Convert a trained neural network from PyTorch via PythonCall.jl to a
[`Pipeline`](@ref).

## Supported layers

 * `nn.Linear`
 * `nn.ReLU`
 * `nn.Sequential`
 * `nn.Sigmoid`
 * `nn.Softplus`
 * `nn.Tanh`

## Keyword arguments

 * `config`: a dictionary that maps `Symbol`s to [`AbstractPredictor`](@ref)s
   that control how the activation functions are reformulated. For example,
   `:Sigmoid => MathOptAI.Sigmoid()` or `:ReLU => MathOptAI.QuadraticReLU()`.
   The supported Symbols are `:ReLU`, `:Sigmoid`, `:SoftPlus`, and `:Tanh`.
 * `gray_box`: if `true`, the neural network is added as a user-defined
   nonlinear operator, with gradients provided by `torch.func.jacrev`.
 * `gray_box_hessian`: if `true`, the gray box additionally computes the Hessian
   of the output using `torch.func.hessian`.
 * `gray_box_device`: device used to construct PyTorch tensors, e.g. `"cuda"`
   to run on an Nvidia GPU.
"""
function MathOptAI.build_predictor(
    predictor::MathOptAI.PytorchModel;
    config::Dict = Dict{Any,Any}(),
    gray_box::Bool = false,
    gray_box_hessian::Bool = false,
    gray_box_device::String = "cpu",
)
    if gray_box
        if !isempty(config)
            error("cannot specify the `config` kwarg if `gray_box = true`")
        end
        return MathOptAI.GrayBox(
            predictor;
            hessian = gray_box_hessian,
            device = gray_box_device,
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
        z = torch.zeros(length(x))
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

function Ipopt.VectorNonlinearOracle(
    predictor::MathOptAI.PytorchModel,
    input_dimension::Int;
    device::String = "cpu",
)
    torch = PythonCall.pyimport("torch")
    torch_model = torch.load(predictor.filename; weights_only = false)
    torch_model = torch_model.to(device)
    J = torch.func.jacrev(torch_model)
    y = torch_model(torch.zeros(input_dimension))
    output_dimension = PythonCall.pyconvert(Int, PythonCall.pybuiltins.len(y))
    # We model the function as:
    #     0 <= f(x) - y <= 0
    function eval_f(ret::AbstractVector, x::AbstractVector)
        py_x = torch.tensor(x[1:input_dimension]; device)
        py_value = torch_model(py_x).detach().cpu().numpy()
        value = PythonCall.pyconvert(Vector, py_value)
        for i in 1:output_dimension
            ret[i] = value[i] - x[input_dimension+i]
        end
        return
    end
    # Note the order of the for-loops, first over the output_dimension, and then
    # across the input_dimension. This makes the Jacobian structure of ∇f(x) be
    # column-major and dense with respect to x.
    jacobian_structure = Tuple{Int64,Int64}[
        (r, c) for c in 1:input_dimension for r in 1:output_dimension
    ]
    # We also need to add non-zero terms for the `-I` component of the Jacobian.
    for i in 1:output_dimension
        push!(jacobian_structure, (i, input_dimension + i))
    end
    function eval_jacobian(ret::AbstractVector, x::AbstractVector)
        py_x = torch.tensor(x[1:input_dimension]; device)
        py_value = J(py_x).detach().cpu().numpy()
        value = PythonCall.pyconvert(Matrix, py_value)
        for i in 1:length(value)
            ret[i] = value[i]             # ∇f(x)
        end
        for i in 1:output_dimension
            ret[length(value)+i] = -1.0   # -I
        end
        return
    end
    # We need to compute only ∇²f(x) because the -y part does not appear in
    # the Hessian.
    #
    # Note the order of the for-loops, first over the rows, and then across the
    # columns, with j >= i ensuring that this is the upper triangle portion of
    # the Hessian-of-the-Lagrangian.
    hessian_lagrangian_structure = Tuple{Int64,Int64}[
        (i, j) for j in 1:input_dimension for i in 1:input_dimension if j >= i
    ]
    lagrangian_layer = torch.nn.Linear(output_dimension, 1, bias = false)
    lagrangian = torch.nn.Sequential(torch_model, lagrangian_layer)
    lagrangian.to(device)
    ∇²L = torch.func.hessian(lagrangian)
    function eval_hessian_lagrangian(ret, x, mult)
        # x contains inputs and outputs
        py_x = torch.tensor(x[1:input_dimension], device = device)
        mult_tensor = torch.tensor([mult], device=device)
        mult_param = torch.nn.Parameter(mult_tensor, requires_grad=false)
        lagrangian[-1].weight = mult_param
        hessian = PythonCall.pyconvert(Array, ∇²L(py_x)[0].detach().cpu().numpy())
        k = 0
        for j in 1:input_dimension
            for i in 1:j
                k += 1
                ret[k] = hessian[i, j]
            end
        end
        return
    end
    return Ipopt.VectorNonlinearOracle(;
        dimension = input_dimension + output_dimension,
        l = zeros(output_dimension),
        u = zeros(output_dimension),
        f = eval_f,
        jacobian_structure,
        jacobian = eval_jacobian,
        hessian_lagrangian_structure,
        hessian_lagrangian = eval_hessian_lagrangian,
    )
end

end  # module
