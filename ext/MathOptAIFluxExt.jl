# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAIFluxExt

import Flux
import JuMP
import MathOptAI
import MathOptInterface as MOI

"""
    MathOptAI.build_predictor(
        predictor::Flux.Chain;
        config::Dict = Dict{Any,Any}(),
        gray_box::Bool = false,
        hessian::Bool = gray_box,
        input_size::Union{Nothing,NTuple{N,Int}} = nothing,
    )

Convert a trained neural network from Flux.jl to a [`Pipeline`](@ref).

## Supported layers

 * `Flux.Conv`
 * `Flux.Dense`
 * `Flux.flatten`
 * `Flux.LayerNorm`
 * `Flux.MaxPool`
 * `Flux.MeanPool`
 * `Flux.Scale`
 * `Flux.softmax`

## Supported activation functions

 * `Flux.relu`
 * `Flux.sigmoid`
 * `Flux.softplus`
 * `Flux.tanh`

## Keyword arguments

 * `config`: see the `Config` section below.

 * `gray_box`: if `true`, the neural network is added using a [`GrayBox`](@ref)
   formulation.

 * `hessian`: if `true`, the `gray_box` formulations compute the Hessian of the
   output using `Flux.hessian`. The default for `hessian` is `true` if
   `gray_box` is used.

 * `input_size`: to disambiguate the input and output sizes of matrix inputs,
   chains containing `Conv`, `LayerNorm`, `MaxPool`, and `MeanPool` layers must
   specify an initial input size.

## Config

The `config` dictionary controls how layers in Flux are mapped to
[`AbstractPredictor`](@ref)s.

Supported keys and and example key-value pairs are:

 * `Flux.MaxPool => (k; kwargs...) -> MathOptAI.MaxPool2dBigM(k; M = 10.0, kwargs...)`
 * `Flux.relu => MathOptAI.ReLU`
 * `Flux.sigmoid => MathOptAI.Sigmoid`
 * `Flux.softmax => MathOptAI.SoftMax`
 * `Flux.softplus => MathOptAI.SoftPlus`
 * `Flux.tanh => MathOptAI.Tanh`

## Example

```jldoctest
julia> using JuMP, MathOptAI, Flux

julia> chain = Flux.Chain(Flux.Dense(1 => 16, Flux.relu), Flux.Dense(16 => 1));

julia> model = Model();

julia> @variable(model, x[1:1]);

julia> y, _ = MathOptAI.add_predictor(
           model,
           chain,
           x;
           config = Dict(Flux.relu => MathOptAI.ReLU),
       );

julia> y
1-element Vector{VariableRef}:
 moai_Affine[1]

julia> MathOptAI.build_predictor(
           chain;
           config = Dict(Flux.relu => MathOptAI.ReLU),
       )
Pipeline with layers:
 * Affine(A, b) [input: 1, output: 16]
 * ReLU()
 * Affine(A, b) [input: 16, output: 1]

julia> MathOptAI.build_predictor(
           chain;
           config = Dict(Flux.relu => MathOptAI.ReLUQuadratic),
       )
Pipeline with layers:
 * Affine(A, b) [input: 1, output: 16]
 * ReLUQuadratic(nothing)
 * Affine(A, b) [input: 16, output: 1]
```
"""
function MathOptAI.build_predictor(
    predictor::Flux.Chain;
    config::Dict = Dict{Any,Any}(),
    gray_box::Bool = false,
    hessian::Bool = gray_box,
    input_size::Union{Nothing,NTuple} = nothing,
)
    if gray_box
        if !isempty(config)
            error("cannot specify the `config` kwarg if `gray_box = true`")
        end
        return MathOptAI.GrayBox(predictor; hessian)
    end
    inner_predictor = MathOptAI.Pipeline(MathOptAI.AbstractPredictor[])
    for layer in predictor.layers
        input_size =
            _build_predictor(inner_predictor, layer, config, input_size)
    end
    return inner_predictor
end

function _build_predictor(
    predictor::MathOptAI.Pipeline,
    layer::Any,
    config::Dict,
    input_size::Any,
)
    try
        p = MathOptAI.build_predictor(layer)
        push!(predictor.layers, p)
        return MathOptAI.output_size(p, input_size)
    catch
        return error("Unsupported layer: $layer")
    end
end

_default(::Any) = missing
_default(::typeof(Flux.relu)) = MathOptAI.ReLU
_default(::typeof(Flux.sigmoid)) = MathOptAI.Sigmoid
_default(::typeof(Flux.softplus)) = MathOptAI.SoftPlus
_default(::typeof(Flux.softmax)) = MathOptAI.SoftMax
_default(::typeof(Flux.tanh)) = MathOptAI.Tanh

function _build_predictor(
    ::MathOptAI.Pipeline,
    ::typeof(identity),
    ::Dict,
    input_size::Union{Nothing,NTuple},
)
    # Do nothing: a linear activation
    return input_size
end

function _build_predictor(
    ::MathOptAI.Pipeline,
    ::typeof(Flux.flatten),
    ::Dict,
    input_size::Union{Nothing,NTuple},
)
    if input_size === nothing
        return nothing
    end
    return (prod(input_size),)
end

function _build_predictor(
    predictor::MathOptAI.Pipeline,
    activation::Function,
    config::Dict,
    input_size::Union{Nothing,NTuple},
)
    layer_fn = get(config, activation, _default(activation))
    if layer_fn === missing
        error("Unsupported activation function: $activation")
    end
    p = layer_fn()
    push!(predictor.layers, p)
    return MathOptAI.output_size(p, input_size)
end

function _build_predictor(
    predictor::MathOptAI.Pipeline,
    layer::Flux.Dense,
    config::Dict,
    input_size::Union{Nothing,Tuple{Int}},
)
    p = MathOptAI.Affine(layer.weight, layer.bias)
    push!(predictor.layers, p)
    input_size = MathOptAI.output_size(p, input_size)
    return _build_predictor(predictor, layer.σ, config, input_size)
end

function _build_predictor(
    predictor::MathOptAI.Pipeline,
    layer::Flux.Scale,
    config::Dict,
    input_size::Union{Nothing,Tuple{Int}},
)
    p = MathOptAI.Scale(layer.scale, layer.bias)
    push!(predictor.layers, p)
    input_size = MathOptAI.output_size(p, input_size)
    return _build_predictor(predictor, layer.σ, config, input_size)
end

function _normalize_input_size(layer, ::Nothing)
    msg = "You must specifiy the `input_size` kwarg when using a layer of type $(typeof(layer))"
    return error(msg)
end

_normalize_input_size(::Any, input_size::NTuple{2,Int}) = (input_size..., 1)
_normalize_input_size(::Any, input_size::NTuple{3,Int}) = input_size

function _build_predictor(
    predictor::MathOptAI.Pipeline,
    layer::Flux.Conv,
    config::Dict,
    input_size::Any,
)
    input_size_normalized = _normalize_input_size(layer, input_size)
    p = MathOptAI.Conv2d(
        layer.weight,
        layer.bias;
        input_size = input_size_normalized,
        padding = layer.pad[1:2],
        stride = layer.stride,
    )
    push!(predictor.layers, p)
    input_size_normalized = MathOptAI.output_size(p, input_size_normalized)
    return _build_predictor(predictor, layer.σ, config, input_size_normalized)
end

function _build_predictor(
    predictor::MathOptAI.Pipeline,
    layer::Flux.MaxPool,
    config::Dict,
    input_size::Any,
)
    input_size_normalized = _normalize_input_size(layer, input_size)
    p = get(config, Flux.MaxPool, MathOptAI.MaxPool2d)(
        layer.k;
        input_size = input_size_normalized,
        padding = layer.pad[1:2],
        stride = layer.stride,
    )::MathOptAI.AbstractPredictor
    push!(predictor.layers, p)
    return MathOptAI.output_size(p, input_size_normalized)
end

function _build_predictor(
    predictor::MathOptAI.Pipeline,
    layer::Flux.MeanPool,
    config::Dict,
    input_size::Any,
)
    input_size_normalized = _normalize_input_size(layer, input_size)
    p = MathOptAI.AvgPool2d(
        layer.k;
        input_size = input_size_normalized,
        padding = layer.pad[1:2],
        stride = layer.stride,
    )
    push!(predictor.layers, p)
    return MathOptAI.output_size(p, input_size_normalized)
end

_weight_and_bias(::Type{T}, f::Flux.Scale, size) where {T} = f.scale, f.bias

function _weight_and_bias(::Type{T}, ::typeof(identity), size) where {T}
    return ones(T, size), zeros(T, size)
end

function _build_predictor(
    predictor::MathOptAI.Pipeline,
    layer::Flux.LayerNorm,
    config::Dict,
    input_size::Any,
)
    input_size_normalized = _normalize_input_size(layer, input_size)
    weight, bias = _weight_and_bias(typeof(layer.ϵ), layer.diag, layer.size)
    p = MathOptAI.LayerNorm(
        layer.size;
        input_size = input_size_normalized,
        eps = layer.ϵ,
        weight,
        bias,
    )
    push!(predictor.layers, p)
    _build_predictor(predictor, layer.λ, config, nothing)
    return MathOptAI.output_size(p, input_size_normalized)
end

function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
    predictor::MathOptAI.GrayBox{<:Flux.Chain},
    x::Vector,
)
    set = _build_set(predictor.predictor, length(x), predictor.hessian)
    y = MathOptAI.add_variables(model, x, set.output_dimension, "moai_Flux")
    con = JuMP.@constraint(model, [x; y] in set)
    return y, MathOptAI.Formulation(predictor, y, [con])
end

function _build_set(chain::Flux.Chain, input_dimension::Int, hessian::Bool)
    output_dimension = only(Flux.outputsize(chain, (input_dimension,)))
    # We model the function as:
    #     0 <= f(x) - y <= 0
    function eval_f(ret::AbstractVector, x::AbstractVector)
        input = Float32.(x[1:input_dimension])
        value = chain(input)
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
        input = Float32.(x[1:input_dimension])
        value = only(Flux.jacobian(chain, input)::Tuple{Matrix{Float32}})
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
    # the output of `chain` that does the multiplication f(x)' * μ.
    #
    # We update the parameters of μ_layer in our `eval_hessian_lagrangian`
    # function.
    μ_layer = Flux.Dense(output_dimension, 1; bias = false)
    L = Flux.Chain(chain, μ_layer)
    function eval_hessian_lagrangian(
        ret::AbstractVector,
        x::AbstractVector,
        μ::AbstractVector,
    )
        input = Float32.(x[1:input_dimension])
        copyto!(μ_layer.weight, μ)
        ∇²L = Flux.hessian(x -> only(L(x)), input)::Matrix{Float32}
        k = 0
        for j in 1:input_dimension
            for i in 1:j
                k += 1
                ret[k] = ∇²L[i, j]
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

end  # module MathOptAIFluxExt
