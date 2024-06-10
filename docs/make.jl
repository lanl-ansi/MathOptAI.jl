# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

import DataFrames
import Documenter
import Flux
import GLM
import Lux
import MathOptAI

Documenter.makedocs(;
    sitename = "MathOptAI.jl",
    authors = "The MathOptAI contributors",
    format = Documenter.HTML(;
        # See https://github.com/JuliaDocs/Documenter.jl/issues/868
        prettyurls = get(ENV, "CI", nothing) == "true",
        collapselevel = 1,
    ),
    pages = ["index.md", "api.md"],
    modules = [
        MathOptAI,
        Base.get_extension(MathOptAI, :MathOptAIFluxExt),
        Base.get_extension(MathOptAI, :MathOptAIGLMExt),
        Base.get_extension(MathOptAI, :MathOptAILuxExt),
        Base.get_extension(MathOptAI, :MathOptAIStatsModelsExt),
    ],
    checkdocs = :exports,
    doctest = true,
)
