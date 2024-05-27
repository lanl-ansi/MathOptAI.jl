# Copyright (c) 2024: Oscar Dowson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    UnivariateNormalDistribution(; mean::Function, std_dev::Function)

A univariate Normal distribution, represented by the functions `mean(x::Vector)`
and `std_dev(x::Vector)`.

## Example

```jldoctest
julia> import Omelette

julia> Omelette.UnivariateNormalDistribution(;
           mean = x -> only(x),
           std_dev = x -> 1.0,
       )
UnivariateNormalDistribution(mean, std_dev)
```
"""
struct UnivariateNormalDistribution{F,G}
    mean::F
    std_dev::G

    function UnivariateNormalDistribution(; mean::Function, std_dev::Function)
        return new{typeof(mean),typeof(std_dev)}(mean, std_dev)
    end
end

function Base.show(io::IO, x::UnivariateNormalDistribution)
    return print(io, "UnivariateNormalDistribution(mean, std_dev)")
end

"""
    add_constraint(
        model::JuMP.Model,
        f::UnivariateNormalDistribution,
        set::MOI.Interval,
        β::Float64,
    )

Add the constraint:
```math
\\mathbb{P}(f(x) \\in [l, u]) \\ge β
```
where \$f(x)~\\mathcal{N}(\\mu, \\sigma)\$ is a normally distributed random
variable given by the `UnivariateNormalDistribution`.

If both `l` and `u` are finite, then the probability mass is equally
distributed, so that each side of the constraint holds with `(1 + β) / 2`.

## Examples

```jldoctest
julia> using JuMP, Omelette

julia> model = Model();

julia> @variable(model, 0 <= x <= 5);

julia> f = Omelette.UnivariateNormalDistribution(;
           mean = x -> only(x),
           std_dev = x -> 1.0,
       );

julia> Omelette.add_constraint(model, f, [x], MOI.Interval(0.5, Inf), 0.95);

julia> print(model)
Feasibility
Subject to
 x ≥ 2.1448536269514715
 x ≥ 0
 x ≤ 5
```
"""
function add_constraint(
    model::JuMP.Model,
    N::UnivariateNormalDistribution,
    x::Vector{JuMP.VariableRef},
    set::MOI.Interval,
    β::Float64,
)
    @assert β >= 0.5
    if isfinite(set.upper) && isfinite(set.lower)
        # Dual-sided chance constraint. In this case, we want β to be the joint
        # probabiltiy, so take an equal probabiltiy each side.
        β = (1 + β) / 2
    end
    if isfinite(set.upper)
        # P(f(x) ≤ u) ≥ β
        # => μ(x) + Φ⁻¹(β) * σ <= u
        λ = Distributions.invlogcdf(Distributions.Normal(0, 1), log(β))
        JuMP.@constraint(model, N.mean(x) + λ * N.std_dev(x) <= set.upper)
    end
    if isfinite(set.lower)
        # P(f(x) ≥ l) ≥ β
        # => μ(x) + Φ⁻¹(1 - β) * σ >= l
        λ = Distributions.invlogcdf(Distributions.Normal(0, 1), log(1 - β))
        JuMP.@constraint(model, N.mean(x) + λ * N.std_dev(x) >= set.lower)
    end
    return
end
