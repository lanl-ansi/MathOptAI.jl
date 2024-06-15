# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAIAbstractGPsExt

import AbstractGPs
import AbstractGPs: StatsBase
import Distributions
import JuMP
import MathOptAI

"""
    MathOptAI.add_predictor(
        model::JuMP.Model,
        predictor::MathOptAI.Quantile{<:AbstractGPs.PosteriorGP},
        x::Vector,
    )
"""
function MathOptAI.add_predictor(
    model::JuMP.Model,
    predictor::MathOptAI.Quantile{<:AbstractGPs.PosteriorGP},
    x::Vector,
)
    M, N = length(x), length(predictor.quantiles)
    function _evaluate(op::Function, distribution, x...)
        input_x = collect(x)
        if M > 1
            input_x = AbstractGPs.RowVecs(reshape(input_x, 1, M))
        end
        return only(op(distribution(input_x)))
    end
    # Mean
    f_mean(x...) = _evaluate(StatsBase.mean, predictor.distribution, x...)
    op_mean = JuMP.add_nonlinear_operator(model, M, f_mean; name = :op_gp_mean)
    μ = JuMP.@variable(model)
    JuMP.@constraint(model, μ == op_mean(x...))
    # Variance
    f_var(x...) = _evaluate(StatsBase.var, predictor.distribution, x...)
    op_var = JuMP.add_nonlinear_operator(model, M, f_var; name = :op_gp_var)
    σ² = JuMP.@variable(model)
    JuMP.@constraint(model, σ² == op_var(x...))
    # Outputs
    y = JuMP.@variable(model, [1:N], base_name = "moai_quantile")
    dist = Distributions.Normal(0, 1)
    λ = Distributions.invlogcdf.(dist, log.(predictor.quantiles))
    JuMP.@constraint(model, y .== μ .+ λ .* sqrt(1e-6 + σ²))
    return y
end

end  # module
