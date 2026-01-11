# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAIStatsModelsExt

import DataFrames
import JuMP
import MathOptAI
import StatsModels

"""
    MathOptAI.add_predictor(
        model::JuMP.AbstractModel,
        predictor::StatsModels.TableRegressionModel,
        x::DataFrames.DataFrame;
        kwargs...,
    )

Add a trained regression model from StatsModels.jl to `model`, using the
DataFrame `x` as input.

In most cases, `predictor` should be a GLM.jl predictor supported by MathOptAI,
but trained using `@formula` and a `DataFrame` instead of the raw matrix input.

In general, `x` may have some columns that are constant (`Float64`) and some
columns that are JuMP decision variables.

## Keyword arguments

All keyword arguments are passed to the corresponding [`add_predictor`](@ref) of
the GLM extension.

## Example

```jldoctest
julia> using DataFrames, GLM, JuMP, MathOptAI

julia> train_df = DataFrames.DataFrame(x1 = rand(10), x2 = rand(10));

julia> train_df.y = 1.0 .* train_df.x1 + 2.0 .* train_df.x2 .+ rand(10);

julia> predictor = GLM.lm(GLM.@formula(y ~ x1 + x2), train_df);

julia> model = Model();

julia> test_df = DataFrames.DataFrame(
           x1 = rand(6),
           x2 = @variable(model, [1:6]),
       );

julia> test_df.y, _ = MathOptAI.add_predictor(model, predictor, test_df);

julia> test_df.y
6-element Vector{VariableRef}:
 moai_Affine[1]
 moai_Affine[1]
 moai_Affine[1]
 moai_Affine[1]
 moai_Affine[1]
 moai_Affine[1]
```
"""
function MathOptAI.add_predictor(
    model::JuMP.AbstractModel,
    predictor::StatsModels.TableRegressionModel,
    df::DataFrames.DataFrame;
    kwargs...,
)
    resp = StatsModels.modelcols(StatsModels.MatrixTerm(predictor.mf.f.rhs), df)
    p = MathOptAI.build_predictor(predictor.model)
    ret = [
        MathOptAI.add_predictor(model, p, collect(row); kwargs...) for
        row in eachrow(resp)
    ]
    formulation = MathOptAI.PipelineFormulation(p, last.(ret))
    return reduce(vcat, first.(ret)), formulation
end

end  # module MathOptAIStatsModelsExt
