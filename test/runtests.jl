# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

import MathOptAI
import ParallelTestRunner

is_test_file(f) = startswith(f, "test_") && endswith(f, ".jl")

testsuite = Dict{String,Expr}()
for (root, dirs, files) in walkdir(@__DIR__)
    for file in joinpath.(root, filter(is_test_file, files))
        testsuite[file] = :(include($file))
    end
end

ParallelTestRunner.runtests(MathOptAI, ARGS; testsuite)
