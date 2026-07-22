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
    core, y, oracle = ExaModels.embed_oracle(
        core,
        x,
        PythonCall.pyconvert(Int, PythonCall.pybuiltins.len(y));
        f! = (ret, x) -> save!(ret, torch_model(torch.tensor(x))),
        jvp!,
        vjp!,
        # From the ExaModels docs: "Use `adapt=Val(true)` to have arrays
        # automatically copied to CPU before each callback invocation.
        adapt = Val(true),
    )
    return (core, y), MathOptAI.Formulation(p, Any[y], Any[oracle])
end

end  # MathOptAIExaModelsPythonCallExt
