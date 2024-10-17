# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module TestStatsModelsExt

using JuMP
using Test

import CSV
import DataFrames
import Downloads
import GLM
import Ipopt
import MathOptAI

is_test(x) = startswith(string(x), "test_")

function runtests()
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

function read_df(filename)
    url = "https://raw.githubusercontent.com/INFORMSJoC/2020.1023/master/data/"
    data = Downloads.download(url * filename)
    return CSV.read(data, DataFrames.DataFrame)
end

function test_student_enrollment()
    historical_df = read_df("college_student_enroll-s1-1.csv")
    model_glm = GLM.glm(
        GLM.@formula(enroll ~ 0 + merit + SAT + GPA),
        historical_df,
        GLM.Bernoulli(),
    )
    application_df = read_df("college_applications6000.csv")
    n_students = size(application_df, 1)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    application_df.merit = @variable(model, 0 <= x_merit[1:n_students] <= 2.5)
    application_df.enroll, _ =
        MathOptAI.add_predictor(model, model_glm, application_df)
    @objective(model, Max, sum(application_df.enroll))
    @constraint(model, sum(application_df.merit) <= 0.2 * n_students)
    optimize!(model)
    @test is_solved_and_feasible(model)
    @test objective_value(model) > 2488
    application_df.merit_sol = value.(application_df.merit)
    application_df.enroll_sol = value.(application_df.enroll)
    @test ≈(sum(application_df.enroll_sol), objective_value(model); atol = 1e-6)
    return
end

end  # module

TestStatsModelsExt.runtests()
