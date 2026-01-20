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
        hessian::Bool = gray_box,
        device::String = "cpu",
        input_size::Union{Nothing,NTuple{N,Int}} = nothing,
    )

Convert a trained neural network from PyTorch via PythonCall.jl to a
[`Pipeline`](@ref).

## Supported layers

 * `nn.AvgPool2d`
 * `nn.Conv2d`
 * `nn.Dropout`
 * `nn.Flatten`
 * `nn.GELU`
 * `nn.LayerNorm`
 * `nn.LeakyReLU`
 * `nn.Linear`
 * `nn.MaxPool2d`
 * `nn.ReLU`
 * `nn.Sequential`
 * `nn.Sigmoid`
 * `nn.Softmax`
 * `nn.Softplus`
 * `nn.Tanh`

Note that `nn.Dropout` layers are skipped because we assume that the PyTorch
model is being evaluated, not trained.

## Keyword arguments

 * `config`: see the `Config` section below.

 * `gray_box`: if `true`, the neural network is added using a [`GrayBox`](@ref)
   formulation.

 * `hessian`: if `true`, the `gray_box` formulation computes the Hessian of the
   output using `torch.func.hessian`. The default for `hessian` is `true` if
   `gray_box` is used.

 * `device`: device used to construct PyTorch tensors, for example, `"cuda"`
   to run on an Nvidia GPU.

 * `input_size`: to disambiguate the input and output sizes of matrix inputs,
   models containing `AvgPool2d`, `Conv2d`, and `MaxPool2d` layers must specify
   an initial input size.

## Config

The `config` dictionary controls how layers in PyTorch are mapped to
[`AbstractPredictor`](@ref)s.

Supported keys and and example key-value pairs are:

 * `:GELU => MathOptAI.GELU`
 * `:MaxPool2d => (k; kwargs...) -> MathOptAI.MaxPool2dBigM(k; M = 10.0, kwargs...)`
 * `:ReLU => MathOptAI.ReLU`
 * `:Sigmoid => MathOptAI.Sigmoid`
 * `:SoftMax => MathOptAI.SoftMax`
 * `:SoftPlus => (; beta) -> MathOptAI.SoftPlus(; beta)`
 * `:Tanh => MathOptAI.Tanh`

Note that `:LeakyReLU` is not supported. Use `:ReLU` to control how the inner
ReLU is modeled.
"""
function MathOptAI.build_predictor(
    predictor::MathOptAI.PytorchModel;
    config::Dict = Dict{Any,Any}(),
    gray_box::Bool = false,
    device::String = "cpu",
    hessian::Bool = gray_box,
    input_size::Union{Nothing,NTuple} = nothing,
)
    if gray_box
        if !isempty(config)
            error("cannot specify the `config` kwarg if `gray_box = true`")
        end
        return MathOptAI.GrayBox(predictor; hessian, device)
    end
    torch = PythonCall.pyimport("torch")
    torch_model = torch.load(predictor.filename; weights_only = false)
    return MathOptAI.build_predictor(torch_model; config, input_size)
end

function _normalize_input_size(p, ::Nothing)
    return error("You must specifiy the `input_size` kwarg when using $p")
end

_normalize_input_size(::Any, input_size::NTuple{2,Int}) = (input_size..., 1)
_normalize_input_size(::Any, input_size::NTuple{3,Int}) = input_size

_to_tuple(p::PythonCall.Py) = _to_tuple(PythonCall.pyconvert(Any, p))
_to_tuple(p::Int) = (p, p)
_to_tuple(p::Tuple{Int,Int}) = p

_is_instance(x, T) = Bool(PythonCall.pybuiltins.isinstance(x, T))

_reversedims(x) = permutedims(x, reverse(1:ndims(x)))

function _weight_and_bias(layer::PythonCall.Py)
    if Bool(layer.elementwise_affine)
        w = _pyconvert(Array{Float64}, layer.weight)
        b = _pyconvert(Array{Float64}, layer.bias)
        return w, b
    end
    shape = PythonCall.pyconvert(Any, layer.normalized_shape)
    return ones(Float64, shape), zeros(Float64, shape)
end

function MathOptAI.build_predictor(
    layer::PythonCall.Py;
    input_size::Union{Nothing,NTuple} = nothing,
    config::Dict = Dict{Any,Any}(),
    nn = PythonCall.pyimport("torch.nn"),
)
    if _is_instance(layer, nn.Sequential)
        layers = MathOptAI.AbstractPredictor[]
        for child in layer
            if _is_instance(child, nn.Dropout)
                continue
            end
            p = MathOptAI.build_predictor(child; config, input_size, nn)
            input_size = MathOptAI.output_size(p, input_size)
            push!(layers, p)
        end
        return MathOptAI.Pipeline(layers)
    elseif _is_instance(layer, nn.Linear)
        return MathOptAI.Affine(
            _pyconvert(Matrix{Float64}, layer.weight),
            _pyconvert(Vector{Float64}, layer.bias),
        )
    elseif _is_instance(layer, nn.AvgPool2d)
        input_size = _normalize_input_size("nn.AvgPool2d", input_size)
        return MathOptAI.AvgPool2d(
            _to_tuple(layer.kernel_size);
            input_size,
            padding = _to_tuple(layer.padding),
            stride = _to_tuple(layer.stride),
        )
    elseif _is_instance(layer, nn.Conv2d)
        w = _pyconvert(Array{Float64,4}, layer.weight)
        w = reverse(permutedims(w, (3, 4, 2, 1)); dims = (1, 2))
        input_size = _normalize_input_size("nn.MaxPool2d", input_size)
        return MathOptAI.Conv2d(
            w,
            _pyconvert(Vector{Float64}, layer.bias);
            input_size,
            padding = _to_tuple(layer.padding),
            stride = _to_tuple(layer.stride),
        )
    elseif _is_instance(layer, nn.Flatten)
        input_size = _normalize_input_size("nn.Flatten", input_size)
        col_major_indices = reshape(1:prod(input_size), input_size)
        p = vec(permutedims(col_major_indices, reverse(1:length(input_size))))
        return MathOptAI.ReducedSpace(MathOptAI.Permutation(p))
    elseif _is_instance(layer, nn.GELU)
        return get(config, :GELU, MathOptAI.GELU)()
    elseif _is_instance(layer, nn.LayerNorm)
        # The layer is complicated because MathOptAI assumes we compute the
        # LayerNorm over [normalized_shape, *], whereas PyTorch assumes we
        # compute the LayerNorm over [*, normalized_shape]
        #
        # First we need to compute a permutation vector to permute the arrays:
        indices = reshape(1:prod(input_size), input_size)
        col_to_row_major = vec(_reversedims(indices))
        row_to_col_major = invperm(col_to_row_major)
        # Now we can create the LayerNorm predictor. Becase we're reversing the
        # dimensions of the input array, we also need to reverse the dimensions
        # of the normalized_shape
        input_size = _normalize_input_size("nn.LayerNorm", reverse(input_size))
        weight, bias = _weight_and_bias(layer)
        return MathOptAI.Pipeline(
            MathOptAI.ReducedSpace(MathOptAI.Permutation(col_to_row_major)),
            get(config, :LayerNorm, MathOptAI.LayerNorm)(
                reverse(PythonCall.pyconvert(Any, layer.normalized_shape));
                input_size,
                eps = PythonCall.pyconvert(Float64, layer.eps),
                weight = _reversedims(weight),
                bias = _reversedims(bias),
            ),
            MathOptAI.ReducedSpace(MathOptAI.Permutation(row_to_col_major)),
        )
    elseif _is_instance(layer, nn.LeakyReLU)
        negative_slope = PythonCall.pyconvert(Float64, layer.negative_slope)
        relu = get(config, :ReLU, MathOptAI.ReLU)()
        return MathOptAI.LeakyReLU(; negative_slope, relu)
    elseif _is_instance(layer, nn.MaxPool2d)
        input_size = _normalize_input_size("nn.MaxPool2d", input_size)
        return get(config, :MaxPool2d, MathOptAI.MaxPool2d)(
            _to_tuple(layer.kernel_size);
            input_size,
            padding = _to_tuple(layer.padding),
            stride = _to_tuple(layer.stride),
        )
    elseif _is_instance(layer, nn.ReLU)
        return get(config, :ReLU, MathOptAI.ReLU)()
    elseif _is_instance(layer, nn.Sigmoid)
        return get(config, :Sigmoid, MathOptAI.Sigmoid)()
    elseif _is_instance(layer, nn.Softmax)
        return get(config, :SoftMax, MathOptAI.SoftMax)()
    elseif _is_instance(layer, nn.Softplus)
        beta = PythonCall.pyconvert(Float64, layer.beta)
        return get(config, :SoftPlus, MathOptAI.SoftPlus)(; beta)
    elseif _is_instance(layer, nn.Tanh)
        return get(config, :Tanh, MathOptAI.Tanh)()
    elseif haskey(config, layer.__class__)
        return config[layer.__class__](layer; input_size, config, nn)
    end
    return error("unsupported layer: $layer")
end

function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
    predictor::MathOptAI.GrayBox{MathOptAI.PytorchModel},
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
