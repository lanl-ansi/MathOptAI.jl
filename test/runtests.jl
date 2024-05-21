# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

using Test

is_test(x) = startswith(x, "test_") && endswith(x, ".jl")

@testset "$file" for file in filter(is_test, readdir(@__DIR__))
    include(joinpath(@__DIR__, file))
end
