# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAIExaModelsPythonCallExt

import ExaModels
import MathOptAI
import PythonCall

_length(x::Union{ExaModels.Variable,ExaModels.Expression}) = x.length

_length(x::AbstractVector) = length(x)

function _pyconvert(::Type{T}, tensor) where {T}
    return PythonCall.pyconvert(T, tensor.detach().cpu().numpy())
end

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.GrayBox{MathOptAI.PytorchModel},
    x::Any;
    kwargs...,
)
    torch = PythonCall.pyimport("torch")
    torch_model = torch.load(p.predictor.filename; weights_only = false)
    torch_model = torch_model.to(p.device)
    y = torch_model(torch.zeros(_length(x); p.device))
    output_dimension = PythonCall.pyconvert(Int, PythonCall.pybuiltins.len(y))
    save!(ret, tensor) = ret .= _pyconvert(Vector, tensor)
    function jvp!(ret, x_in, v_in)
        x, v = torch.tensor(x_in), torch.tensor(v_in)
        _, jvp = torch.func.jvp(torch_model, (x,), (v,))
        return save!(ret, jvp)
    end
    function vjp!(ret, x, w)
        _, vjp_fn = torch.func.vjp(torch_model, torch.tensor(x))
        return save!(ret, vjp_fn(torch.tensor(w)))
    end
    # We want to compute hvp(x, μ, v) = (sum_i μ_i ∇²f_i(x)) * v
    # Using L(x, μ) = μ' * f(x), we have:
    #   hvp(x, μ, v) = ∇²L(x, μ) * v
    #                = ∇(∇L(x, μ)) * v
    # which is a Jacobian-vector product of the gradient of L
    # In AD terms, this is forward-over-reverse.
    μ_layer = torch.nn.Linear(output_dimension, 1; bias = false)
    L = torch.nn.Sequential(torch_model, μ_layer).to(p.device)
    function hvp!(ret, x_in, μ_in, v_in)
        py_μ = torch.tensor([μ_in]; p.device)
        μ_layer.weight = torch.nn.Parameter(py_μ; requires_grad = false)
        x, v = torch.tensor(x_in), torch.tensor(v_in)
        _, hvp = torch.func.jvp(torch.func.grad(x -> L(x)[0]), (x,), (v,))
        return save!(ret, hvp)
    end
    core, y, oracle = ExaModels.embed_oracle(
        core,
        x,
        output_dimension;
        f! = (ret, x) -> save!(ret, torch_model(torch.tensor(x))),
        jvp!,
        vjp!,
        hvp!,
        # From the ExaModels docs: "Use `adapt=Val(true)` to have arrays
        # automatically copied to CPU before each callback invocation.
        adapt = Val(true),
    )
    return (core, y), MathOptAI.Formulation(p, Any[y], Any[oracle])
end

end  # MathOptAIExaModelsPythonCallExt
