# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAIFluxExt

import Flux
import JuMP
import MathOptAI

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
   `Ipopt._VectorNonlinearOracle`. This is an experimental feature that may
   offer better performance than `gray_box`. To use this feature, you MUST use
   Ipopt as the optimizer.

 * `hessian`: if `true`, the `gray_box` and `vector_nonlinear_oracle`
   formulations compute the Hessian of the output using `Flux.hessian`.
   The default for `hessian` is `false` if `gray_box` is used, and `true` if
   `vector_nonlinear_oracle` is used.

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

end  # module
