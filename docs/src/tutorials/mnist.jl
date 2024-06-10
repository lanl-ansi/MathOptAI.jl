# Copyright (c) 2024: Oscar Dowson and contributors                         #src
# Copyright (c) 2024: Triad National Security, LLC                          #src
#                                                                           #src
# Use of this source code is governed by a BSD-style license that can be    #src
# found in the LICENSE.md file.                                             #src

# # Adversarial machine learning with Flux.jl

# The purpose of this tutorial is to explain how to embed a neural network model
# from [Flux.jl](https://github.com/FluxML/Flux.jl) into JuMP.

# ## Required packages

# This tutorial requires the following packages

using JuMP
import Flux
import Ipopt
import MathOptAI
import MLDatasets
import Plots
import Statistics

# ## Data

# This tutorial uses images from the MNIST dataset.

# We load the predefined train and test splits:

train_data = MLDatasets.MNIST(; split = :train)
test_data = MLDatasets.MNIST(; split = :test)

# Since the data are images, it is helpful to plot them. (This requires a
# transpose and reversing the rows to get the orientation correct.)

function plot_image(instance)
    return Plots.heatmap(
        instance.features'[28:-1:1, :];
        title = "Label = $(instance.targets)",
        xlims = (1, 28),
        ylims = (1, 28),
        aspect_ratio = true,
        legend = false,
        xaxis = false,
        yaxis = false,
    )
end

Plots.plot(
    [plot_image(train_data[i]) for i in 1:6]...;
    layout = (2, 3),
)

# ## Training

# We use a simple neural network with one hidden layer and a sigmoid activation
# function. (There are better performing networks; try experimenting.)

ml_model = Flux.Chain(
    Flux.Dense(28^2 => 32, Flux.sigmoid),
    Flux.Dense(32 => 10),
    Flux.softmax,
)

# Then, we use [Flux.jl](https://github.com/FluxML/Flux.jl) to train our model.

# !!! note
#     It is not the purpose of this tutorial to explain how Flux works; see the
#     documentation at [https://fluxml.ai](https://fluxml.ai) for more details.

begin
    x2dim = reshape(train_data.features, 28^2, :)
    yhot = Flux.onehotbatch(train_data.targets, 0:9)
    train_loader = Flux.DataLoader(
        (x2dim, yhot);
        batchsize = 256,
        shuffle = true,
    )
    opt_state = Flux.setup(Flux.Adam(3e-4), ml_model)
    ## Run only 30 epochs. You can improve the loss by traing further.
    for epoch in 1:30
        loss = 0.0
        for (x, y) in train_loader
            l, gs = Flux.withgradient(m -> Flux.crossentropy(m(x), y), ml_model)
            Flux.update!(opt_state, ml_model, gs[1])
            loss += l / length(train_loader)
        end
        println("Epoch $epoch: loss = $loss")
    end
end

# Let's have a look at some of the predictions:

function plot_image(ml_model, x::Matrix)
    predictions = ml_model(Float32.(vec(x)))
    score, index = findmax(predictions)
    score_per = round(Int, 100 * score)
    return Plots.heatmap(
        x'[28:-1:1, :];
        title = "Predicted: $(index - 1) ($(score_per)%)",
        xlims = (1, 28),
        ylims = (1, 28),
        aspect_ratio = true,
        legend = false,
        xaxis = false,
        yaxis = false,
    )
end

Plots.plot(
    [plot_image(ml_model, test_data[i].features) for i in 1:16]...,
    size = (1200, 1200),
)

# ## JuMP

# Now that we have a trained machine learning model, we can embed it in a JuMP
# model.

# Here's a function which takes a test case and returns an example that
# maximizes the probability of the adversarial example.

function find_adversarial_image(test_case; adversary_label, δ = 0.05)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, 0 <= x[1:28, 1:28] <= 1)
    @constraint(model, -δ .<= x .- test_case.features .<= δ)
    y = MathOptAI.add_predictor(model, ml_model, vec(x))
    @objective(model, Max, y[adversary_label+1] - y[test_case.targets+1])
    optimize!(model)
    @assert is_solved_and_feasible(model)
    return value.(x)
end

# Let's try finding an adversarial example to the third test image. The image on
# the left is our input image. The network this this is a `1` with probability
# 99%. The image on the right is the adversarial image. The network this this is
# a `7`, although it is less confident.

x_adversary = find_adversarial_image(test_data[3]; adversary_label = 7);
Plots.plot(
    plot_image(ml_model, test_data[3].features),
    plot_image(ml_model, x_adversary)
)
