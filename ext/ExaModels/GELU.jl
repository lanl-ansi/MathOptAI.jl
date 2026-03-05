# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

# GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / π) * (x + 0.044715 * x^3)))
_gelu(x) = 0.5 * x * (1.0 + tanh(sqrt(2 / π) * (x + 0.044715 * x^3)))

# Starting from:
#   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / π) * (x + 0.044715 * x^3)))
# Make the substitution:
#   u = sqrt(2 / π) * (x + 0.044715 * x^3)
# Then:
#   GELU(x) = 0.5 * x * (1 + tanh(u))
# By the product rule:
#   GELU'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * tanh(u)'
# and
#   tanh(u)' = (1 - tanh(u)^2) * u'
# and
#   u' = sqrt(2 / π) * (1 + 3 * 0.044715 * x^2)
# so
#   GELU'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * (1 - tanh(u)^2) * u'
function _d_gelu(x)
    a = sqrt(2 / π)
    u = a * (x + 0.044715 * x^3)
    d_u = a * (1 + 3 * 0.044715 * x^2)
    tanh_u = tanh(u)
    return 0.5 * (1 + tanh_u) + 0.5 * x * (1 - tanh_u^2) * d_u
end

# Starting from:
#   u = sqrt(2 / π) * (x + 0.044715 * x^3)
#   v = (1 - tanh(u)^2)
#   GELU'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * v * u'
# which has two additive terms:
#   GELU'(x) = A + B
# where
#   A = 0.5 * (1 + tanh(u))
#   B = 0.5 * x * v * u'
# The first part is easy
#   u' = sqrt(2 / π) * (1 + 3 * 0.044715 * x^2)
#   A' = 0.5 * tanh(u)' = 0.5 * v * u'
# The other part is a bit harder
#   B' = (0.5 * x) * (v * u')' + 0.5 * v * u'
#      = (0.5 * x) * (v * u')' + A'
#      = (0.5 * x) * (v * u'' + v' * u') + A'
# GELU''(x) = 0.5 * v * u' + (0.5 * x) * (v * u'' + v' * u') + 0.5 * v * u'
#           = v * u' + (0.5 * x) * (v * u'' + v' * u')
function _dd_gelu(x)
    a = sqrt(2 / π)
    u = a * (x + 0.044715 * x^3)
    d_u = a * (1 + 3 * 0.044715 * x^2)
    dd_u = a * 6 * 0.044715 * x
    tanh_u = tanh(u)
    v = 1 - tanh_u^2
    d_v = -2 * tanh_u * v * d_u
    return v * d_u + 0.5 * x * (v * dd_u + d_v * d_u)
end

ExaModels.@register_univariate(_gelu, _d_gelu, _dd_gelu)

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.GELU,
    x::ExaModels.AbstractVariable,
)
    n = _length(x)
    y = ExaModels.variable(core, n)
    c1 = ExaModels.constraint(
        core,
        y[i] - _gelu(x[i]) for i in 1:n;
        lcon = 0.0,
        ucon = 0.0,
    )
    return y, MathOptAI.Formulation(p, Any[y], Any[c1])
end

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.GELU,
    x::AbstractVector,
)
    n = length(x)
    y = ExaModels.variable(core, n)
    cons = [
        ExaModels.constraint(core, y[i] - _gelu(x[i]); lcon = 0.0, ucon = 0.0) for i in 1:n
    ]
    return y, MathOptAI.Formulation(p, Any[y], cons)
end

function MathOptAI.add_predictor(
    ::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.GELU},
    x,
)
    y = [_gelu(x[i]) for i in 1:_length(x)]
    return y, MathOptAI.Formulation(p)
end
