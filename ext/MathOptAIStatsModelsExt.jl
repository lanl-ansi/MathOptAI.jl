# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module MathOptAIStatsModelsExt

import DataFrames
import JuMP
import MathOptAI
import StatsModels

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
