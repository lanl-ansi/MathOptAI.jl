# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

import AbstractGPs
import DataFrames
import DecisionTree
import Documenter
import EvoTrees
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
#  Modify the release notes
# ==============================================================================

function fix_release_line(
    line::String,
    url::String = "https://github.com/lanl-ansi/MathOptAI.jl",
)
    # (#XXXX) -> ([#XXXX](url/issue/XXXX))
    while (m = match(r"\(\#([0-9]+)\)", line)) !== nothing
        id = m.captures[1]
        line = replace(line, m.match => "([#$id]($url/issues/$id))")
    end
    # ## Version X.Y.Z -> [Version X.Y.Z](url/releases/tag/vX.Y.Z)
    while (m = match(r"\#\# Version ([0-9]+.[0-9]+.[0-9]+)", line)) !== nothing
        tag = m.captures[1]
        line = replace(
            line,
            m.match => "## [Version $tag]($url/releases/tag/v$tag)",
        )
    end
    # ## vX.Y.Z -> [vX.Y.Z](url/releases/tag/vX.Y.Z)
    while (m = match(r"\#\# (v[0-9]+.[0-9]+.[0-9]+)", line)) !== nothing
        tag = m.captures[1]
        line = replace(line, m.match => "## [$tag]($url/releases/tag/$tag)")
    end
    return line
end

function _fix_release_lines(changelog, release_notes, args...)
    open(release_notes, "w") do io
        for line in readlines(changelog; keep = true)
            write(io, fix_release_line(line, args...))
        end
    end
    return
end

_fix_release_lines(
    joinpath(@__DIR__, "src", "changelog.md"),
    joinpath(@__DIR__, "src", "release_notes.md"),
)

function _add_edit_url(filename, url)
    contents = read(filename, String)
    open(filename, "w") do io
        write(io, "```@meta\nEditURL = \"$url\"\n```\n\n")
        write(io, contents)
        return
    end
    return
end

_add_edit_url(joinpath(@__DIR__, "src", "release_notes.md"), "changelog.md")

# ==============================================================================
#  Build the documentation
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
            "manual/EvoTrees.md",
            "manual/Flux.md",
            "manual/GLM.md",
            "manual/Lux.md",
            "manual/PythonCall.md",
            "manual/PyTorch.md",
        ],
        "Tutorials" => [
            "tutorials/student_enrollment.md",
            "tutorials/decision_trees.md",
            "tutorials/mnist.md",
            "tutorials/mnist_lux.md",
            "tutorials/pytorch.md",
            "tutorials/gaussian.md",
            "tutorials/graph_neural_networks.md",
        ],
        "Developers" =>
            ["developers/checklists.md", "developers/design_principles.md"],
        "api.md",
        "release_notes.md",
    ],
    modules = [
        MathOptAI,
        Base.get_extension(MathOptAI, :MathOptAIAbstractGPsExt),
        Base.get_extension(MathOptAI, :MathOptAIEvoTreesExt),
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

# ==============================================================================
#  Deploy everything in `build`
# ==============================================================================

Documenter.deploydocs(;
    repo = "github.com/lanl-ansi/MathOptAI.jl.git",
    push_preview = true,
)
