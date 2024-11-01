# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

using Test

import Documenter
import MathOptAI

is_test(x) = startswith(x, "test_") && endswith(x, ".jl")

@testset "$file" for file in filter(is_test, readdir(@__DIR__))
    include(joinpath(@__DIR__, file))
end

@testset "Docstrings" begin
    Documenter.doctest(MathOptAI; manual = false)
end
