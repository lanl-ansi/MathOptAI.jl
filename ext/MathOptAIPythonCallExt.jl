# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAIPythonCallExt

import JuMP
import PythonCall
import MathOptAI
import MathOptInterface as MOI

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

 * `nn.GELU`
 * `nn.LeakyReLU`
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
   The supported Symbols are:
    * `:GELU`
    * `:ReLU`
    * `:Sigmoid`
    * `:SoftMax`
    * `:SoftPlus`
    * `:Tanh`
    Note that `:LeakyReLU` is not supported. Use `:ReLU` to control how the
    inner ReLU is modeled.

 * `gray_box`: if `true`, the neural network is added using a [`GrayBox`](@ref)
   formulation.

 * `vector_nonlinear_oracle`: if `true`, the neural network is added using
   a [`VectorNonlinearOracle`](@ref) formulation.

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
    elseif Bool(PythonCall.pybuiltins.isinstance(layer, nn.GELU))
        return get(config, :GELU, MathOptAI.GELU())
    elseif Bool(PythonCall.pybuiltins.isinstance(layer, nn.LeakyReLU))
        negative_slope = PythonCall.pyconvert(Float64, layer.negative_slope)
        relu = get(config, :ReLU, MathOptAI.ReLU())
        return MathOptAI.LeakyReLU(; negative_slope, relu)
    elseif Bool(PythonCall.pybuiltins.isinstance(layer, nn.ReLU))
        return get(config, :ReLU, MathOptAI.ReLU())
    elseif Bool(PythonCall.pybuiltins.isinstance(layer, nn.Sequential))
        layers = [_predictor(nn, child, config) for child in layer]
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

function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
    predictor::MathOptAI.VectorNonlinearOracle{MathOptAI.PytorchModel},
    x::Vector,
)
    set = _build_set(
        predictor.predictor.filename,
        length(x),
        predictor.device,
        predictor.hessian,
    )
    y = MathOptAI.add_variables(model, x, set.output_dimension, "moai_Pytorch")
    con = JuMP.@constraint(model, [x; y] in set)
    return y, MathOptAI.Formulation(predictor, y, [con])
end

function _pyconvert(::Type{T}, tensor) where {T}
    return PythonCall.pyconvert(T, tensor.detach().cpu().numpy())
end

function _build_set(
    filename::String,
    input_dimension::Int,
    device::String,
    hessian::Bool,
)
    torch = PythonCall.pyimport("torch")
    torch_model = torch.load(filename; weights_only = false)
    torch_model = torch_model.to(device)
    y = torch_model(torch.zeros(input_dimension; device))
    output_dimension = PythonCall.pyconvert(Int, PythonCall.pybuiltins.len(y))
    # We model the function as:
    #     0 <= f(x) - y <= 0
    function eval_f(ret::AbstractVector, x::AbstractVector)
        py_x = torch.tensor(x[1:input_dimension]; device)
        value = _pyconvert(Vector, torch_model(py_x))
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
    J = torch.func.jacrev(torch_model)
    function eval_jacobian(ret::AbstractVector, x::AbstractVector)
        py_x = torch.tensor(x[1:input_dimension]; device)
        value = _pyconvert(Matrix, J(py_x))
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
    # We want to compute the Hessian-of-the-Lagrangian:
    #   ∇²L(x) = Σ μᵢ ∇²fᵢ(x)
    # We could compute this by calculating the
    # `output_dimension * input_dimension * input_dimension` dense hessian and
    # then sum over the first dimension multiplying by μᵢ. This is pretty slow.
    #
    # Instead, we define L(x) = Σ μᵢ fᵢ(x), and then compute a single dense
    # Hessian ∇²L(x).
    #
    # We can define L(x) by adding a new output_dimension-by-1 linear layer to
    # the output of `torch_model` that does the multiplication f(x)' * μ.
    #
    # We update the parameters of μ_layer in our `eval_hessian_lagrangian`
    # function.
    μ_layer = torch.nn.Linear(output_dimension, 1; bias = false)
    L = torch.nn.Sequential(torch_model, μ_layer)
    L.to(device)
    ∇²L = torch.func.hessian(L)
    function eval_hessian_lagrangian(
        ret::AbstractVector,
        x::AbstractVector,
        μ::AbstractVector,
    )
        py_x = torch.tensor(x[1:input_dimension]; device)
        # [μ] is Python syntax to make a 1xN tensor. Even though μ_layer is Nx1,
        # torch.nn.Linear stores the weight matrix transposed.
        py_μ = torch.tensor([μ]; device)
        μ_layer.weight = torch.nn.Parameter(py_μ; requires_grad = false)
        value = _pyconvert(Matrix, ∇²L(py_x)[0])
        k = 0
        for j in 1:input_dimension
            for i in 1:j
                k += 1
                ret[k] = value[i, j]
            end
        end
        return
    end
    return MOI.VectorNonlinearOracle(;
        dimension = input_dimension + output_dimension,
        l = zeros(output_dimension),
        u = zeros(output_dimension),
        eval_f,
        jacobian_structure,
        eval_jacobian,
        hessian_lagrangian_structure,
        eval_hessian_lagrangian = ifelse(
            hessian,
            eval_hessian_lagrangian,
            nothing,
        ),
    )
end

end  # module MathOptAIPythonCallExt
