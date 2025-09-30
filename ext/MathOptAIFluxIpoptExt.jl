# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAIFluxIpoptExt

import Flux
import Ipopt
import JuMP
import MathOptAI

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
    return Ipopt._VectorNonlinearOracle(;
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

end  # module MathOptAIFluxIpoptExt
