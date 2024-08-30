# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAIAbstractGPsExt

import AbstractGPs
import Distributions
import JuMP
import MathOptAI

"""
    MathOptAI.add_predictor(
        model::JuMP.AbstractModel,
        predictor::MathOptAI.Quantile{<:AbstractGPs.PosteriorGP},
        x::Vector,
    )

Add the quantiles of a trained Gaussian Process from AbstractGPs.jl to `model`.

## Example

```jldoctest
julia> using JuMP, MathOptAI, AbstractGPs

julia> x_data = 2π .* (0.0:0.1:1.0);

julia> y_data = sin.(x_data);

julia> fx = AbstractGPs.GP(AbstractGPs.Matern32Kernel())(x_data, 0.1);

julia> p_fx = AbstractGPs.posterior(fx, y_data);

julia> model = Model();

julia> @variable(model, 1 <= x[1:1] <= 6, start = 3);

julia> predictor = MathOptAI.Quantile(p_fx, [0.1, 0.9]);

julia> y, _ = MathOptAI.add_predictor(model, predictor, x);

julia> y
2-element Vector{VariableRef}:
 moai_quantile[1]
 moai_quantile[2]

julia> @objective(model, Max, y[2] - y[1])
moai_quantile[2] - moai_quantile[1]
```
"""
function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
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
    f_mean(x...) = _evaluate(AbstractGPs.mean, predictor.distribution, x...)
    op_mean = JuMP.add_nonlinear_operator(model, M, f_mean; name = :op_gp_mean)
    μ = JuMP.@variable(model)
    JuMP.@constraint(model, μ == op_mean(x...))
    # Variance
    f_var(x...) = _evaluate(AbstractGPs.var, predictor.distribution, x...)
    op_var = JuMP.add_nonlinear_operator(model, M, f_var; name = :op_gp_var)
    σ² = JuMP.@variable(model, start = 1)
    JuMP.@constraint(model, σ² == op_var(x...))
    # Outputs
    dist = Distributions.Normal(0, 1)
    λ = Distributions.invlogcdf.(dist, log.(predictor.quantiles))
    y = JuMP.@variable(model, [1:N], base_name = "moai_quantile")
    JuMP.set_start_value.(y, λ)
    JuMP.@constraint(model, y .== μ .+ λ .* sqrt(σ²))
    return y, MathOptAI.Formulation(predictor)
end

end  # module
