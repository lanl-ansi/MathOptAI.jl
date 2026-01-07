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
        vector_nonlinear_oracle::Bool = false,
        hessian::Bool = vector_nonlinear_oracle,
    )

Convert a trained neural network from Flux.jl to a [`Pipeline`](@ref).

## Supported layers

 * `Flux.Dense`
 * `Flux.Scale`
 * `Flux.softmax`

## Supported activation functions

 * `Flux.relu`
 * `Flux.sigmoid`
 * `Flux.softplus`
 * `Flux.tanh`

## Keyword arguments

 * `config`: a dictionary that maps supported `Flux` activation functions to
   [`AbstractPredictor`](@ref)s that control how the activation functions are
   reformulated. For example, `Flux.sigmoid => MathOptAI.Sigmoid()` or
   `Flux.relu => MathOptAI.QuadraticReLU()`.

 * `gray_box`: if `true`, the neural network is added using a [`GrayBox`](@ref)
   formulation.

 * `vector_nonlinear_oracle`: if `true`, the neural network is added using
   a [`VectorNonlinearOracle`](@ref) formulation.

 * `hessian`: if `true`, the `gray_box` and `vector_nonlinear_oracle`
   formulations compute the Hessian of the output using `Flux.hessian`.
   The default for `hessian` is `false` if `gray_box` is used, and `true` if
   `vector_nonlinear_oracle` is used.

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
           config = Dict(Flux.relu => MathOptAI.ReLU()),
       );

julia> y
1-element Vector{VariableRef}:
 moai_Affine[1]

julia> MathOptAI.build_predictor(
           chain;
           config = Dict(Flux.relu => MathOptAI.ReLU()),
       )
Pipeline with layers:
 * Affine(A, b) [input: 1, output: 16]
 * ReLU()
 * Affine(A, b) [input: 16, output: 1]

julia> MathOptAI.build_predictor(
           chain;
           config = Dict(Flux.relu => MathOptAI.ReLUQuadratic()),
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
    vector_nonlinear_oracle::Bool = false,
    hessian::Bool = vector_nonlinear_oracle,
    # For backwards compatibility
    gray_box_hessian::Bool = false,
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
        )
    end
    if gray_box
        if !isempty(config)
            error("cannot specify the `config` kwarg if `gray_box = true`")
        end
        return MathOptAI.GrayBox(
            predictor;
            hessian = hessian | gray_box_hessian,
        )
    end
    inner_predictor = MathOptAI.Pipeline(MathOptAI.AbstractPredictor[])
    for layer in predictor.layers
        _build_predictor(inner_predictor, layer, config)
    end
    return inner_predictor
end

function MathOptAI.GrayBox(predictor::Flux.Chain; hessian::Bool = false)
    function output_size(x)
        return only(Flux.outputsize(predictor, (length(x),)))
    end
    function callback(x)
        x32 = collect(Float32.(x))
        ret = Flux.withjacobian(predictor, x32)
        if !hessian
            return (value = ret.val, jacobian = only(ret.grad))
        end
        Hs = map(1:length(ret.val)) do i
            return Flux.hessian(x -> predictor(x)[i], x32)
        end
        H = cat(Hs...; dims = 3)
        return (value = ret.val, jacobian = only(ret.grad), hessian = H)
    end
    return MathOptAI.GrayBox(output_size, callback; has_hessian = hessian)
end

function _build_predictor(::MathOptAI.Pipeline, layer::Any, ::Dict)
    return error("Unsupported layer: $layer")
end

_default(::typeof(identity)) = nothing
_default(::Any) = missing
_default(::typeof(Flux.relu)) = MathOptAI.ReLU()
_default(::typeof(Flux.sigmoid)) = MathOptAI.Sigmoid()
_default(::typeof(Flux.softplus)) = MathOptAI.SoftPlus()
_default(::typeof(Flux.softmax)) = MathOptAI.SoftMax()
_default(::typeof(Flux.tanh)) = MathOptAI.Tanh()

function _build_predictor(
    predictor::MathOptAI.Pipeline,
    activation::Function,
    config::Dict,
)
    layer = get(config, activation, _default(activation))
    if layer === nothing
        # Do nothing: a linear activation
    elseif layer === missing
        error("Unsupported activation function: $activation")
    else
        push!(predictor.layers, layer)
    end
    return
end

function _build_predictor(
    predictor::MathOptAI.Pipeline,
    layer::Flux.Dense,
    config::Dict,
)
    push!(predictor.layers, MathOptAI.Affine(layer.weight, layer.bias))
    _build_predictor(predictor, layer.σ, config)
    return
end

function _build_predictor(
    predictor::MathOptAI.Pipeline,
    layer::Flux.Scale,
    config::Dict,
)
    push!(predictor.layers, MathOptAI.Scale(layer.scale, layer.bias))
    _build_predictor(predictor, layer.σ, config)
    return
end

function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
    predictor::MathOptAI.VectorNonlinearOracle{<:Flux.Chain},
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
