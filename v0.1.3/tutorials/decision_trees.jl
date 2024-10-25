# Copyright (c) 2024: Oscar Dowson and contributors                         #src
# Copyright (c) 2024: Triad National Security, LLC                          #src
#                                                                           #src
# Use of this source code is governed by a BSD-style license that can be    #src
# found in the LICENSE.md file.                                             #src

# # Classification problems with DecisionTree.jl

# The purpose of this tutorial is to explain how to embed a decision tree
# model from [DecisionTree.jl](https://github.com/JuliaAI/DecisionTree.jl) into
# JuMP.

# The data and example in this tutorial comes from the paper: David Bergman,
# Teng Huang, Philip Brooks, Andrea Lodi, Arvind U. Raghunathan (2021) JANOS: An
# Integrated Predictive and Prescriptive Modeling Framework. INFORMS Journal on
# Computing 34(2):807-816.
# [https://doi.org/10.1287/ijoc.2020.1023](https://doi.org/10.1287/ijoc.2020.1023)

# ## Required packages

# This tutorial uses the following packages.

using JuMP
import CSV
import DataFrames
import Downloads
import DecisionTree
import HiGHS
import MathOptAI
import Statistics

# ## Data

# Here is a function to load the data directly from the JANOS repository:

function read_df(filename)
    url = "https://raw.githubusercontent.com/INFORMSJoC/2020.1023/master/data/"
    data = Downloads.download(url * filename)
    return CSV.read(data, DataFrames.DataFrame)
end

# There are two important files. The first, `college_student_enroll-s1-1.csv`,
# contains historial admissions data on anonymized students, their SAT score,
# their GPA, their merit scholarships, and whether the enrolled in the college.

train_df = read_df("college_student_enroll-s1-1.csv")

# The second, `college_applications6000.csv`, contains the SAT and GPA data of
# students who are currently applying:

evaluate_df = read_df("college_applications6000.csv")

# There are 6,000 prospective students:

n_students = size(evaluate_df, 1)

# ## Prediction model

# The first step is to train a logistic regression model to predict the Boolean
# `enroll` column based on the `SAT`, `GPA`, and `merit` columns.

train_features = Matrix(train_df[:, [:SAT, :GPA, :merit]])
train_labels = train_df[:, :enroll]
predictor = DecisionTree.DecisionTreeClassifier(; max_depth = 3)
DecisionTree.fit!(predictor, train_features, train_labels)
DecisionTree.print_tree(predictor)

# ## Decision model

# Now that we have a trained decision tree, we want a decision model that
# chooses the optimal merit scholarship for each student in:

evaluate_df

# Here's an empty JuMP model to start:

model = Model()

# First, we add a new columnn to `evaluate_df`, with one JuMP decision variable
# for each row. It is important the the `.merit` column name in `evaluate_df`
# matches the name in `train_df`.

evaluate_df.merit = @variable(model, 0 <= x_merit[1:n_students] <= 2.5);
evaluate_df

# Then, we use [`MathOptAI.add_predictor`](@ref) to embed `model_ml` into the
# JuMP `model`. [`MathOptAI.add_predictor`](@ref) returns a vector of variables,
# one for each row in `evaluate_df`, corresponding to the output `enroll` of
# our logistic regression.

evaluate_features = Matrix(evaluate_df[:, [:SAT, :GPA, :merit]])
evaluate_df.enroll = mapreduce(vcat, 1:size(evaluate_features, 1)) do i
    y, _ = MathOptAI.add_predictor(model, predictor, evaluate_features[i, :])
    return y
end
evaluate_df

# The `.enroll` column name in `evaluate_df` is just a name. It doesn't have to
# match the name in `train_df`.

# The objective of our problem is to maximize the expected number of students
# who enroll:

@objective(model, Max, sum(evaluate_df.enroll))

# Subject to the constraint that we can spend at most `0.2 * n_students` on
# merit scholarships:

@constraint(model, sum(evaluate_df.merit) <= 0.2 * n_students)

# Because logistic regression involves a [`Sigmoid`](@ref) layer, we need to use
# a smooth nonlinear optimizer. A common choice is Ipopt. Solve and check the
# optimizer found a feasible solution:

set_optimizer(model, HiGHS.Optimizer)
set_silent(model)
optimize!(model)
@assert is_solved_and_feasible(model)

# Let's store the solution in `evaluate_df` for analysis:

evaluate_df.merit_sol = value.(evaluate_df.merit);
evaluate_df.enroll_sol = value.(evaluate_df.enroll);
evaluate_df

# ## Solution analysis

# We expect that just under 2,500 students will enroll:

sum(evaluate_df.enroll_sol)

# We awarded merit scholarships to approximately 1 in 6 students:

count(evaluate_df.merit_sol .> 1e-5)

# The average merit scholarship was worth just under \$1,000:

1_000 * Statistics.mean(evaluate_df.merit_sol[evaluate_df.merit_sol.>1e-5])
