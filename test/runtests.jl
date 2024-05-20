# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

using Test

for file in readdir(joinpath(@__DIR__, "models"))
    if startswith(file, "test_") && endswith(file, ".jl")
        @testset "$file" begin
            include(joinpath(@__DIR__, "models", file))
        end
    end
end
