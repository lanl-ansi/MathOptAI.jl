# PyTorch

[PyTorch](https://pytorch.org) is a library for machine learning in Python.

MathOptAI supports embedding PyTorch models in JuMP if they are composed of:

 * `nn.Linear`
 * `nn.ReLU`
 * `nn.Sequential`
 * `nn.Sigmoid`
 * `nn.Tanh`

Here is an example:

```julia
julia> using JuMP, MathOptAI

julia> model = Model()

julia> @variable(model, x[1:2]);

julia> predictor = MathOptAI.PytorchModel('saved_pytorch_model.pt');

julia> y, _ = MathOptAI.add_predictor(model, predictor, x);
```
