# Copyright (c) 2024: Oscar Dowson and contributors
# Copyright (c) 2024: Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

import AbstractGPs
import DataFrames
import DecisionTree
import Documenter
import Flux
import GLM
import Literate
import Lux
import MathOptAI
import PythonCall
import Test

# ==============================================================================
#  Literate
# ==============================================================================

function _file_list(full_dir, relative_dir, extension)
    return map(
        file -> joinpath(relative_dir, file),
        filter(file -> endswith(file, extension), sort(readdir(full_dir))),
    )
end

function _include_sandbox(filename)
    mod = @eval module $(gensym()) end
    return Base.include(mod, filename)
end

function _literate_directory(dir)
    for filename in _file_list(dir, dir, ".md")
        rm(filename)
    end
    for filename in _file_list(dir, dir, ".jl")
        # `include` the file to test it before `#src` lines are removed. It is
        # in a testset to isolate local variables between files.
        Test.@testset "$(filename)" begin
            _include_sandbox(filename)
        end
        Literate.markdown(filename, dir; documenter = true)
    end
    # Convert `@example` blocks into `@repl` blocks in the following files:
    for file in ["student_enrollment.md", "decision_trees.md"]
        filename = joinpath(@__DIR__, "src", "tutorials", file)
        content = read(filename, String)
        content = replace(content, "@example" => "@repl")
        write(filename, content)
    end
    return
end

_literate_directory(joinpath(@__DIR__, "src", "tutorials"))

# ==============================================================================
#  makedocs
# ==============================================================================

Documenter.makedocs(;
    sitename = "MathOptAI.jl",
    authors = "The MathOptAI contributors",
    format = Documenter.HTML(;
        # See https://github.com/JuliaDocs/Documenter.jl/issues/868
        prettyurls = get(ENV, "CI", nothing) == "true",
        collapselevel = 1,
    ),
    pages = [
        "index.md",
        "Manual" => [
            "manual/predictors.md",
            "manual/AbstractGPs.md",
            "manual/DecisionTree.md",
            "manual/Flux.md",
            "manual/GLM.md",
            "manual/Lux.md",
            "manual/PyTorch.md",
        ],
        "Tutorials" => [
            "tutorials/student_enrollment.md",
            "tutorials/decision_trees.md",
            "tutorials/mnist.md",
            "tutorials/mnist_lux.md",
            "tutorials/pytorch.md",
            "tutorials/gaussian.md",
        ],
        "Developers" => ["developers/design_principles.md"],
        "api.md",
    ],
    modules = [
        MathOptAI,
        Base.get_extension(MathOptAI, :MathOptAIAbstractGPsExt),
        Base.get_extension(MathOptAI, :MathOptAIDecisionTreeExt),
        Base.get_extension(MathOptAI, :MathOptAIFluxExt),
        Base.get_extension(MathOptAI, :MathOptAIGLMExt),
        Base.get_extension(MathOptAI, :MathOptAILuxExt),
        Base.get_extension(MathOptAI, :MathOptAIPythonCallExt),
        Base.get_extension(MathOptAI, :MathOptAIStatsModelsExt),
    ],
    checkdocs = :exports,
    doctest = true,
)
