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

julia> f = Omelette.Pipeline([
           Omelette.LinearLayer([1.0 2.0], [0.0]),
           Omelette.ReLUQuadratic(1),
       ])
Omelette.Pipeline(Omelette.AbstractPredictor[Omelette.LinearLayer([1.0 2.0], [0.0]), Omelette.ReLUQuadratic(1)])

julia> y = Omelette.add_predictor(model, f, x)
1-element Vector{VariableRef}:
 omelette_y[1]

julia> print(model)
Feasibility
Subject to
 -x[1] - 2 x[2] + omelette_y[1] = 0
 omelette_y[1] - _z[1]+ + _z[1]- = 0
 _z[1]+*_z[1]- = 0
 _z[1]+ ≥ 0
 _z[1]- ≥ 0
```
"""
struct Pipeline <: AbstractPredictor
    layers::Vector{AbstractPredictor}
end

Base.size(x::Pipeline) = (size(last(x.layers), 1), size(first(x.layers), 2))

function _add_predictor_inner(
    model::JuMP.Model,
    predictor::Pipeline,
    x::Vector{JuMP.VariableRef},
    y::Vector{JuMP.VariableRef},
)
    for (i, layer) in enumerate(predictor.layers)
        if i == length(predictor.layers)
            add_predictor!(model, layer, x, y)
        else
            x = add_predictor(model, layer, x)
        end
    end
    return
end
