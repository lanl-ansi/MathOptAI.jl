# Copyright (c) 2024: Triad National Security, LLC
# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptAIExaModelsExt

import ExaModels
import MathOptAI

# ── Helpers ──────────────────────────────────────────────────────────────────

_exa_length(x::ExaModels.Variable) = x.length
_exa_length(x::ExaModels.Subexpr) = x.length
_exa_length(x::ExaModels.ReducedSubexpr) = x.length
_exa_length(x::AbstractVector) = length(x)

# ── Activation function registrations ────────────────────────────────────────
# tanh, max, min, exp, log are already registered in ExaModels — no action needed.
# ReLU uses max(0, x) directly.

_moai_exa_sigmoid(x) = inv(one(x) + exp(-x))
_moai_exa_d_sigmoid(x) = (s = _moai_exa_sigmoid(x); s * (one(s) - s))
function _moai_exa_dd_sigmoid(x)
    return (s = _moai_exa_sigmoid(x); s * (one(s) - s) * (one(s) - 2s))
end
ExaModels.@register_univariate(
    _moai_exa_sigmoid,
    _moai_exa_d_sigmoid,
    _moai_exa_dd_sigmoid,
)

# GELU: 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
const _MOAI_C1 = sqrt(2 / π)
const _MOAI_C2 = 0.044715

_moai_exa_gelu(x) = 0.5x * (1 + tanh(_MOAI_C1 * (x + _MOAI_C2 * x^3)))

function _moai_exa_d_gelu(x)
    z = _MOAI_C1 * (x + _MOAI_C2 * x^3)
    t = tanh(z)
    dz = _MOAI_C1 * (1 + 3 * _MOAI_C2 * x^2)
    return 0.5 * (1 + t) + 0.5 * x * (1 - t^2) * dz
end

function _moai_exa_dd_gelu(x)
    z = _MOAI_C1 * (x + _MOAI_C2 * x^3)
    t = tanh(z)
    dz = _MOAI_C1 * (1 + 3 * _MOAI_C2 * x^2)
    d2z = _MOAI_C1 * 6 * _MOAI_C2 * x
    dt = 1 - t^2
    return dt * dz + 0.5 * dt * dz + 0.5 * x * (-2t * dt * dz^2 + dt * d2z)
end

ExaModels.@register_univariate(
    _moai_exa_gelu,
    _moai_exa_d_gelu,
    _moai_exa_dd_gelu,
)

# SoftPlus: log(1 + exp(β*x)) / β
# We use exp, log which are already registered; SoftPlus is expressed inline.

# ── Affine (full-space) ───────────────────────────────────────────────────────

"""
    MathOptAI.add_predictor(core::ExaModels.ExaCore, p::MathOptAI.Affine, x)

Add an [`Affine`](@ref) predictor to an `ExaCore`.

Each column of `A` is stored as an `ExaModels.Parameter` and added to the base
constraint `y[i] - b[i] = 0` via `constraint!`, producing a single GPU-friendly
`SIMDFunction` kernel per column.
"""
function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.Affine,
    x,
)
    A, b = p.A, p.b
    m, n = size(A)
    y = ExaModels.variable(core, m)
    b_param = ExaModels.parameter(core, b)
    # Base: y[i] - b[i] = 0, parallelized over i ∈ 1:m
    c1 = ExaModels.constraint(
        core,
        y[i] - b_param[i] for i in 1:m;
        lcon = 0.0,
        ucon = 0.0,
    )
    # Augment column-by-column: subtract A[i,j]*x[j] from constraint i.
    # Full expression: y[i] - b[i] - sum_j A[i,j]*x[j] = 0  ⟺  y[i] = b[i] + A*x
    # x[j] with a fixed integer j produces a fixed Var node — GPU-friendly.
    for j in 1:n
        A_col = ExaModels.parameter(core, A[:, j])
        xj = x[j]
        ExaModels.constraint!(core, c1, i => -A_col[i] * xj for i in 1:m)
    end
    return y, MathOptAI.Formulation(p, [y], Any[c1])
end

# ── ReducedSpace{Affine} ──────────────────────────────────────────────────────

"""
    MathOptAI.add_predictor(
        core::ExaModels.ExaCore,
        p::MathOptAI.ReducedSpace{<:MathOptAI.Affine},
        x,
    )

Add a reduced-space [`Affine`](@ref) predictor to an `ExaCore`.

Returns a `Vector{AbstractNode}` of symbolic expressions without adding new
variables or constraints.
"""
function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.Affine},
    x,
)
    A, b = p.predictor.A, p.predictor.b
    y = [
        b[i] + sum(
            A[i, j] * x[j] for j in axes(A, 2) if !iszero(A[i, j]);
            init = zero(eltype(b)),
        ) for i in axes(A, 1)
    ]
    return y, MathOptAI.Formulation(p)
end

# ── Scale (full-space) ────────────────────────────────────────────────────────

"""
    MathOptAI.add_predictor(core::ExaModels.ExaCore, p::MathOptAI.Scale, x)

Add a [`Scale`](@ref) predictor to an `ExaCore`.

Element-wise: `y[i] = scale[i]*x[i] + bias[i]`.
"""
function MathOptAI.add_predictor(core::ExaModels.ExaCore, p::MathOptAI.Scale, x)
    n = _exa_length(x)
    s_param = ExaModels.parameter(core, p.scale)
    b_param = ExaModels.parameter(core, p.bias)
    y = ExaModels.variable(core, n)
    c1 = ExaModels.constraint(
        core,
        y[i] - s_param[i] * x[i] - b_param[i] for i in 1:n;
        lcon = 0.0,
        ucon = 0.0,
    )
    return y, MathOptAI.Formulation(p, [y], Any[c1])
end

# ── ReducedSpace{Scale} ───────────────────────────────────────────────────────

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.Scale},
    x,
)
    s, b = p.predictor.scale, p.predictor.bias
    y = [s[i] * x[i] + b[i] for i in 1:_exa_length(x)]
    return y, MathOptAI.Formulation(p)
end

# ── Pipeline (full-space) ─────────────────────────────────────────────────────

"""
    MathOptAI.add_predictor(core::ExaModels.ExaCore, p::MathOptAI.Pipeline, x)

Add a [`Pipeline`](@ref) predictor to an `ExaCore` by chaining `add_predictor`
calls for each layer.
"""
function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.Pipeline,
    x,
)
    form = MathOptAI.PipelineFormulation(p, Any[])
    for layer in p.layers
        x, inner = MathOptAI.add_predictor(core, layer, x)
        push!(form.layers, inner)
    end
    return x, form
end

# ── ReducedSpace{Pipeline} ────────────────────────────────────────────────────

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.Pipeline},
    x,
)
    form = MathOptAI.PipelineFormulation(p, Any[])
    for layer in p.predictor.layers
        x, inner =
            MathOptAI.add_predictor(core, MathOptAI.ReducedSpace(layer), x)
        push!(form.layers, inner)
    end
    return x, form
end

# ── ReLU (full-space, GPU-friendly for AbstractVariable inputs) ───────────────

"""
    MathOptAI.add_predictor(
        core::ExaModels.ExaCore,
        p::MathOptAI.ReLU,
        x::ExaModels.AbstractVariable,
    )

Add a [`ReLU`](@ref) predictor to an `ExaCore` using a single generator-based
constraint (GPU-friendly when `x` is a `Variable` or `Subexpr`).
"""
function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ReLU,
    x::ExaModels.AbstractVariable,
)
    n = _exa_length(x)
    y = ExaModels.variable(core, n; lvar = 0.0)
    c1 = ExaModels.constraint(
        core,
        y[i] - max(0, x[i]) for i in 1:n;
        lcon = 0.0,
        ucon = 0.0,
    )
    return y, MathOptAI.Formulation(p, [y], Any[c1])
end

# Scalar fallback for Vector{AbstractNode} (e.g., output of a ReducedSpace layer).
# One constraint per element — correct but not GPU-vectorized.
function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ReLU,
    x::AbstractVector,
)
    n = length(x)
    y = ExaModels.variable(core, n; lvar = 0.0)
    cons = [
        ExaModels.constraint(core, y[i] - max(0, x[i]); lcon = 0.0, ucon = 0.0) for i in 1:n
    ]
    return y, MathOptAI.Formulation(p, [y], cons)
end

# ── ReducedSpace{ReLU} ────────────────────────────────────────────────────────

function MathOptAI.add_predictor(
    ::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.ReLU},
    x,
)
    y = [max(0, x[i]) for i in 1:_exa_length(x)]
    return y, MathOptAI.Formulation(p)
end

# ── LeakyReLU (full-space, GPU-friendly for AbstractVariable inputs) ─────────

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.LeakyReLU,
    x::ExaModels.AbstractVariable,
)
    y_relu, f_relu = MathOptAI.add_predictor(core, p.relu, x)
    n = _exa_length(x)
    η = p.negative_slope
    y = ExaModels.variable(core, n)
    c1 = ExaModels.constraint(
        core,
        y[i] - η * x[i] - (1 - η) * y_relu[i] for i in 1:n;
        lcon = 0.0,
        ucon = 0.0,
    )
    return y,
    MathOptAI.Formulation(p, [f_relu.variables; y], [f_relu.constraints; c1])
end

# Scalar fallback for Vector{AbstractNode}.
function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.LeakyReLU,
    x::AbstractVector,
)
    y_relu, f_relu = MathOptAI.add_predictor(core, p.relu, x)
    n = length(x)
    η = p.negative_slope
    y = ExaModels.variable(core, n)
    cons = [
        ExaModels.constraint(
            core,
            y[i] - η * x[i] - (1 - η) * y_relu[i];
            lcon = 0.0,
            ucon = 0.0,
        ) for i in 1:n
    ]
    return y,
    MathOptAI.Formulation(
        p,
        [f_relu.variables; y],
        [f_relu.constraints; cons...],
    )
end

# ── ReducedSpace{LeakyReLU} ──────────────────────────────────────────────────

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.LeakyReLU},
    x,
)
    inner = p.predictor
    y_relu, _ =
        MathOptAI.add_predictor(core, MathOptAI.ReducedSpace(inner.relu), x)
    η = inner.negative_slope
    y = [η * x[i] + (1 - η) * y_relu[i] for i in 1:_exa_length(x)]
    return y, MathOptAI.Formulation(p)
end

# ── Permutation (ReducedSpace only) ─────────────────────────────────────────

function MathOptAI.add_predictor(
    ::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{MathOptAI.Permutation},
    x,
)
    perm = p.predictor.p
    y = [x[perm[i]] for i in eachindex(perm)]
    return y, MathOptAI.Formulation(p)
end

# ── SoftMax (full-space, GPU-friendly for AbstractVariable inputs) ───────────

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.SoftMax,
    x::ExaModels.AbstractVariable,
)
    n = _exa_length(x)
    denom = ExaModels.variable(core, 1; lvar = 0.0)
    y = ExaModels.variable(core, n; lvar = 0.0, uvar = 1.0)
    # denom[1] - sum_j exp(x[j]) = 0
    c_denom = ExaModels.constraint(
        core,
        denom[1] for i in 1:1;
        lcon = 0.0,
        ucon = 0.0,
    )
    for j in 1:n
        xj = x[j]
        ExaModels.constraint!(core, c_denom, i => -exp(xj) for i in 1:1)
    end
    # y[i] - exp(x[i]) / denom[1] = 0
    d = denom[1]
    c_y = ExaModels.constraint(
        core,
        y[i] - exp(x[i]) / d for i in 1:n;
        lcon = 0.0,
        ucon = 0.0,
    )
    return y, MathOptAI.Formulation(p, [denom, y], Any[c_denom, c_y])
end

# Scalar fallback for Vector{AbstractNode}.
function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.SoftMax,
    x::AbstractVector,
)
    n = length(x)
    denom = ExaModels.variable(core, 1; lvar = 0.0)
    y = ExaModels.variable(core, n; lvar = 0.0, uvar = 1.0)
    denom_expr = denom[1]
    for j in 1:n
        denom_expr = denom_expr - exp(x[j])
    end
    c_denom = ExaModels.constraint(core, denom_expr; lcon = 0.0, ucon = 0.0)
    d = denom[1]
    cons_y = [
        ExaModels.constraint(
            core,
            y[i] - exp(x[i]) / d;
            lcon = 0.0,
            ucon = 0.0,
        ) for i in 1:n
    ]
    return y, MathOptAI.Formulation(p, [denom, y], Any[c_denom; cons_y...])
end

# ── ReducedSpace{SoftMax} ───────────────────────────────────────────────────

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{MathOptAI.SoftMax},
    x,
)
    n = _exa_length(x)
    denom = ExaModels.variable(core, 1; lvar = 0.0)
    c_denom = ExaModels.constraint(
        core,
        denom[1] for i in 1:1;
        lcon = 0.0,
        ucon = 0.0,
    )
    for j in 1:n
        xj = x[j]
        ExaModels.constraint!(core, c_denom, i => -exp(xj) for i in 1:1)
    end
    d = denom[1]
    y = [exp(x[j]) / d for j in 1:n]
    return y, MathOptAI.Formulation(p, [denom], Any[c_denom])
end

# ── Sigmoid (full-space) ──────────────────────────────────────────────────────

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.Sigmoid,
    x::ExaModels.AbstractVariable,
)
    n = _exa_length(x)
    y = ExaModels.variable(core, n; lvar = 0.0, uvar = 1.0)
    c1 = ExaModels.constraint(
        core,
        y[i] - _moai_exa_sigmoid(x[i]) for i in 1:n;
        lcon = 0.0,
        ucon = 0.0,
    )
    return y, MathOptAI.Formulation(p, [y], Any[c1])
end

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.Sigmoid,
    x::AbstractVector,
)
    n = length(x)
    y = ExaModels.variable(core, n; lvar = 0.0, uvar = 1.0)
    cons = [
        ExaModels.constraint(
            core,
            y[i] - _moai_exa_sigmoid(x[i]);
            lcon = 0.0,
            ucon = 0.0,
        ) for i in 1:n
    ]
    return y, MathOptAI.Formulation(p, [y], cons)
end

# ── ReducedSpace{Sigmoid} ─────────────────────────────────────────────────────

function MathOptAI.add_predictor(
    ::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.Sigmoid},
    x,
)
    y = [_moai_exa_sigmoid(x[i]) for i in 1:_exa_length(x)]
    return y, MathOptAI.Formulation(p)
end

# ── Tanh (full-space) ─────────────────────────────────────────────────────────
# tanh is already registered in ExaModels.

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.Tanh,
    x::ExaModels.AbstractVariable,
)
    n = _exa_length(x)
    y = ExaModels.variable(core, n; lvar = -1.0, uvar = 1.0)
    c1 = ExaModels.constraint(
        core,
        y[i] - tanh(x[i]) for i in 1:n;
        lcon = 0.0,
        ucon = 0.0,
    )
    return y, MathOptAI.Formulation(p, [y], Any[c1])
end

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.Tanh,
    x::AbstractVector,
)
    n = length(x)
    y = ExaModels.variable(core, n; lvar = -1.0, uvar = 1.0)
    cons = [
        ExaModels.constraint(core, y[i] - tanh(x[i]); lcon = 0.0, ucon = 0.0) for i in 1:n
    ]
    return y, MathOptAI.Formulation(p, [y], cons)
end

# ── ReducedSpace{Tanh} ────────────────────────────────────────────────────────

function MathOptAI.add_predictor(
    ::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.Tanh},
    x,
)
    y = [tanh(x[i]) for i in 1:_exa_length(x)]
    return y, MathOptAI.Formulation(p)
end

# ── SoftPlus (full-space) ─────────────────────────────────────────────────────
# Expressed inline using exp and log (already registered in ExaModels).

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.SoftPlus,
    x::ExaModels.AbstractVariable,
)
    n = _exa_length(x)
    β = p.beta
    y = ExaModels.variable(core, n; lvar = 0.0)
    c1 = ExaModels.constraint(
        core,
        y[i] - log(1 + exp(β * x[i])) / β for i in 1:n;
        lcon = 0.0,
        ucon = 0.0,
    )
    return y, MathOptAI.Formulation(p, [y], Any[c1])
end

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.SoftPlus,
    x::AbstractVector,
)
    n = length(x)
    β = p.beta
    y = ExaModels.variable(core, n; lvar = 0.0)
    cons = [
        ExaModels.constraint(
            core,
            y[i] - log(1 + exp(β * x[i])) / β;
            lcon = 0.0,
            ucon = 0.0,
        ) for i in 1:n
    ]
    return y, MathOptAI.Formulation(p, [y], cons)
end

# ── ReducedSpace{SoftPlus} ────────────────────────────────────────────────────

function MathOptAI.add_predictor(
    ::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.SoftPlus},
    x,
)
    β = p.predictor.beta
    y = [log(1 + exp(β * x[i])) / β for i in 1:_exa_length(x)]
    return y, MathOptAI.Formulation(p)
end

# ── GELU (full-space) ─────────────────────────────────────────────────────────

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.GELU,
    x::ExaModels.AbstractVariable,
)
    n = _exa_length(x)
    y = ExaModels.variable(core, n)
    c1 = ExaModels.constraint(
        core,
        y[i] - _moai_exa_gelu(x[i]) for i in 1:n;
        lcon = 0.0,
        ucon = 0.0,
    )
    return y, MathOptAI.Formulation(p, [y], Any[c1])
end

function MathOptAI.add_predictor(
    core::ExaModels.ExaCore,
    p::MathOptAI.GELU,
    x::AbstractVector,
)
    n = length(x)
    y = ExaModels.variable(core, n)
    cons = [
        ExaModels.constraint(
            core,
            y[i] - _moai_exa_gelu(x[i]);
            lcon = 0.0,
            ucon = 0.0,
        ) for i in 1:n
    ]
    return y, MathOptAI.Formulation(p, [y], cons)
end

# ── ReducedSpace{GELU} ────────────────────────────────────────────────────────

function MathOptAI.add_predictor(
    ::ExaModels.ExaCore,
    p::MathOptAI.ReducedSpace{<:MathOptAI.GELU},
    x,
)
    y = [_moai_exa_gelu(x[i]) for i in 1:_exa_length(x)]
    return y, MathOptAI.Formulation(p)
end

# ── GrayBox (unsupported) ─────────────────────────────────────────────────────

function MathOptAI.add_predictor(
    ::ExaModels.ExaCore,
    ::MathOptAI.GrayBox,
    ::Any,
)
    return error(
        "GrayBox is not supported with ExaCore. Convert your model to a " *
        "Pipeline of explicit layer predictors.",
    )
end

end  # module MathOptAIExaModelsExt
