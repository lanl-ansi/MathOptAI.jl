# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module MathOptAIStatsModelsExt

import DataFrames
import JuMP
import MathOptAI
import StatsModels

"""
    MathOptAI.add_predictor(
        model::JuMP.Model,
        predictor::StatsModels.TableRegressionModel,
        x::DataFrames.DataFrame,
    )

Add a trained regression model from StatsModels.jl to `model`, using the
DataFrame `x` as input.

In most cases, `predictor` should be a GLM.jl predictor supported by MathOptAI,
but trained using `@formula` and a `DataFrame` instead of the raw matrix input.

In general, `x` may have some columns that are constant (`Float64`) and some
columns that are JuMP decision variables.

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

julia> test_df.y = MathOptAI.add_predictor(model, predictor, test_df)
6-element Vector{VariableRef}:
 omelette_Affine[1]
 omelette_Affine[1]
 omelette_Affine[1]
 omelette_Affine[1]
 omelette_Affine[1]
 omelette_Affine[1]
```
"""
function MathOptAI.add_predictor(
    model::JuMP.Model,
    predictor::StatsModels.TableRegressionModel,
    x::DataFrames.DataFrame,
)
    resp = StatsModels.modelcols(StatsModels.MatrixTerm(predictor.mf.f.rhs), x)
    y = MathOptAI.add_predictor(model, predictor.model, Matrix(resp'))
    @assert size(y, 1) == 1
    return reshape(y, length(y))
end

end  # module