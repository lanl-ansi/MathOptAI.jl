# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

function MathOptAI.add_predictor(
    ::ExaModels.ExaCore,
    ::MathOptAI.GrayBox,
    ::Any,
)
    return error(
        "GrayBox is not supported with ExaCore. Convert your model to a " *
        "Pipeline of explicit layer predictors.",
    )
end
