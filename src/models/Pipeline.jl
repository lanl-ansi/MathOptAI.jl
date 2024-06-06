# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    Pipeline(layers::Vector{AbstractPredictor})

A pipeline of nested layers
```math
f(x) = l_N(\\ldots(l_2(l_1(x))
```

## Example

```jldoctest
julia> using JuMP, Omelette

julia> model = Model();

julia> @variable(model, x[1:2]);

julia> f = Omelette.Pipeline(
           Omelette.LinearRegression([1.0 2.0], [0.0]),
           Omelette.ReLUQuadratic(),
       )
Omelette.Pipeline(Omelette.AbstractPredictor[Omelette.LinearRegression([1.0 2.0], [0.0]), Omelette.ReLUQuadratic()])

julia> y = Omelette.add_predictor(model, f, x)
1-element Vector{VariableRef}:
 omelette_y[1]

julia> print(model)
Feasibility
Subject to
 x[1] + 2 x[2] - omelette_y[1] = 0
 omelette_y[1] - omelette_y[1] + _z[1] = 0
 omelette_y[1]*_z[1] = 0
 omelette_y[1] ≥ 0
 _z[1] ≥ 0
```
"""
struct Pipeline <: AbstractPredictor
    layers::Vector{AbstractPredictor}
end

Pipeline(args::AbstractPredictor...) = Pipeline(collect(args))

function add_predictor(
    model::JuMP.Model,
    predictor::Pipeline,
    x::Vector{JuMP.VariableRef},
)
    for layer in predictor.layers
        x = add_predictor(model, layer, x)
    end
    return x
end
