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
    predictor::MathOptAI.GrayBox{MathOptAI.PytorchModel},
    x::Any;
    kwargs...,
)
    device = predictor.device
    torch = PythonCall.pyimport("torch")
    torch_model = torch.load(predictor.predictor.filename; weights_only = false)
    torch_model = torch_model.to(device)
    save!(ret, tensor) = ret .= _pyconvert(Vector, tensor)
    function jvp(x_in, v_in)
        x, v = torch.tensor(x_in), torch.tensor(v_in)
        _, jvp = torch.func.jvp(torch_model, x, v)
        return jvp
    end
    function vjp(x, w)
        _, vjp_fn = torch.func.vjp(torch_model, torch.tensor(x))
        return vjp_fn(torch.tensor(w))
    end
    core, y, oracle = ExaModels.embed_oracle(
        core,
        x,
        _length(x);
        f! = (ret, x) -> save!(ret, model(torch.tensor(x))),
        jvp! = (ret, x, v) -> save!(ret, jvp(x, v)),
        vjp! = (ret, x, w) -> save!(ret, jvp(x, w)),
        # From the ExaModels docs: "Use `adapt=Val(true)` to have arrays
        # automatically copied to CPU before each callback invocation.
        adapt = Val(true),
    )
    return (core, y), MathOptAI.Formulation(p, Any[y], Any[oracle])
end

end  # MathOptAIExaModelsPythonCallExt
