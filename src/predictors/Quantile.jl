# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

struct Quantile{D} <: AbstractPredictor
    distribution::D
    quantiles::Vector{Float64}
end
