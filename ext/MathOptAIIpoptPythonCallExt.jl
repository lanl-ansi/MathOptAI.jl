# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAIIpoptPythonCallExt

import JuMP
import Ipopt
import PythonCall
import MathOptAI

"""
    MathOptAI.add_predictor(
        model::JuMP.AbstractModel,
        predictor::MathOptAI.VectorNonlinearOracle{MathOptAI.PytorchModel},
        x::Vector;
        device::String = "cpu",
    )

Add a trained neural network from PyTorch via PythonCall.jl to `model`.

## Keyword arguments

 * `device`: device used to construct PyTorch tensors, e.g. `"cuda"` to run on
   an Nvidia GPU.
"""
function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
    predictor::MathOptAI.VectorNonlinearOracle{MathOptAI.PytorchModel},
    x::Vector;
    device::String = "cpu",
)
    set = _build_set_and_callback(predictor.predictor, length(x); device)
    y = JuMP.@variable(model, [1:set.output_dimension])
    con = JuMP.@constraint(model, [x; y] in set)
    return y, MathOptAI.Formulation(predictor, y, [con])
end

function _build_set_and_callback(
    predictor::MathOptAI.PytorchModel,
    input_dimension::Int;
    device::String,
)
    torch = PythonCall.pyimport("torch")
    torch_model = torch.load(predictor.filename; weights_only = false)
    torch_model = torch_model.to(device)
    J = torch.func.jacrev(torch_model)
    y = torch_model(torch.zeros(input_dimension; device))
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
        py_hessian = ∇²L(py_x)[0].detach().cpu().numpy()
        hessian = PythonCall.pyconvert(Array, py_hessian)
        k = 0
        for j in 1:input_dimension
            for i in 1:j
                k += 1
                ret[k] = hessian[i, j]
            end
        end
        return
    end
    return Ipopt._VectorNonlinearOracle(;
        dimension = input_dimension + output_dimension,
        l = zeros(output_dimension),
        u = zeros(output_dimension),
        eval_f,
        jacobian_structure,
        eval_jacobian,
        hessian_lagrangian_structure,
        eval_hessian_lagrangian,
    )
end

end  # module
