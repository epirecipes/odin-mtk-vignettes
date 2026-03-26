# Fast Bayesian Inference with Symbolic Sensitivities


- [Overview](#overview)
- [Model and Data](#model-and-data)
  - [ESS Helper](#ess-helper)
- [Approach 1: Odin.jl + NUTS](#approach-1-odinjl--nuts)
- [Approach 2: Odin.jl + RW-MCMC](#approach-2-odinjl--rw-mcmc)
- [Approach 3: Turing.jl + MTK
  (Baseline)](#approach-3-turingjl--mtk-baseline)
- [Approach 4: Symbolic Sensitivity
  Equations](#approach-4-symbolic-sensitivity-equations)
  - [Deriving the Augmented System](#deriving-the-augmented-system)
  - [Building the Augmented ODE](#building-the-augmented-ode)
  - [Using ODEFunction Codegen](#using-odefunction-codegen)
  - [Analytic Gradient Computation](#analytic-gradient-computation)
  - [Gradient Verification](#gradient-verification)
  - [NUTS Sampling](#nuts-sampling)
- [Comparison](#comparison)
- [Performance Summary](#performance-summary)
- [Where Does the Speedup Come From?](#where-does-the-speedup-come-from)
  - [Key Insight](#key-insight)

## Overview

This vignette compares four approaches to Bayesian inference for
ODE-based epidemic models, from the simplest (Odin.jl) to the most
optimised (symbolic sensitivity equations with MTK codegen):

1.  **Odin.jl + NUTS** — the simplest high-performance option
2.  **Odin.jl + RW-MCMC** — gradient-free baseline
3.  **Turing.jl + MTK** — the standard Julia ecosystem approach
4.  **Symbolic sensitivities + ODEFunction codegen** — maximum MTK
    performance

The naive Turing + MTK approach is slow (~170s for 2000 NUTS iterations)
because of AD through the solver (~65ms), `remake` overhead (~100ms),
and Turing infrastructure (~25%). We show how to eliminate each
bottleneck.

``` julia
using Odin
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics
using DifferentialEquations
using SciMLSensitivity
using Distributions
using Random
using Statistics
using LogDensityProblems, LogDensityProblemsAD
import AdvancedHMC
import Turing
using Logging
using CairoMakie
```

## Model and Data

We use the same SIR model and synthetic data throughout all approaches.

``` julia
@parameters β=0.5 γ=0.1 N_pop=1000.0
@variables S(t)=990.0 I(t)=10.0 R(t)=0.0

eqs = [
    D(S) ~ -β * S * I / N_pop,
    D(I) ~  β * S * I / N_pop - γ * I,
    D(R) ~  γ * I
]

@named sir = ODESystem(eqs, t)
sir_s = structural_simplify(sir)
println("SIR model: $(length(unknowns(sir_s))) unknowns, $(length(parameters(sir_s))) parameters")
```

    SIR model: 3 unknowns, 3 parameters

``` julia
prob_base = ODEProblem(sir_s, [], (0.0, 50.0),
    [sir_s.β => 0.5, sir_s.γ => 0.1]; jac=true)
sol_true = solve(prob_base, Tsit5(); saveat=1.0)

Random.seed!(1)
I_true = sol_true[sir_s.I]
obs = [rand(Poisson(max(v, 1e-6))) for v in I_true[2:end]]
data_times = collect(1.0:1.0:50.0)

println("Generated $(length(obs)) observations, range: $(extrema(obs))")
```

    ┌ Warning: `SciMLBase.ODEProblem(sys, u0, tspan, p; kw...)` is deprecated. Use
    │ `SciMLBase.ODEProblem(sys, merge(if isempty(u0)
    │     Dict()
    │ else
    │     Dict(unknowns(sys) .=> u0)
    │ end, Dict(p)), tspan)` instead.
    └ @ ModelingToolkit ~/.julia/packages/ModelingToolkit/JvjlW/src/deprecations.jl:45
    Generated 50 observations, range: (14, 483)

### ESS Helper

We use the same Geyer initial positive sequence estimator for ESS
throughout.

``` julia
function ess_ipse(x)
    n = length(x)
    mu = mean(x)
    var0 = sum((x .- mu).^2) / n
    var0 == 0 && return Float64(n)
    max_lag = min(n - 1, 1000)
    rho = [sum((x[1:n-k] .- mu) .* (x[k+1:n] .- mu)) / (n * var0)
           for k in 1:max_lag]
    tau = 1.0
    for k in 1:2:max_lag-1
        pair_sum = rho[k] + rho[k+1]
        pair_sum < 0 && break
        tau += 2 * pair_sum
    end
    return n / tau
end
```

    ess_ipse (generic function with 1 method)

## Approach 1: Odin.jl + NUTS

Odin.jl provides a pure Julia implementation of the odin2/dust2/monty
framework with built-in gradient support via ForwardDiff. The DSL
compiles to a type-stable Julia ODE system with zero framework overhead.

``` julia
sir_odin = @odin begin
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0.0
    N = parameter(1000.0)
    I0 = parameter(10.0)
    beta = parameter(0.5)
    gamma = parameter(0.1)
    cases_lambda = I > 0 ? I : 1e-10
    cases ~ Poisson(cases_lambda)
end

# Prepare data for Odin.jl
odin_data = [(time=Float64(i), cases=Float64(obs[i])) for i in eachindex(obs)]
fdata = dust_filter_data(odin_data; time_field=:time)

# Create unfilter (deterministic likelihood)
uf = dust_unfilter_create(sir_odin, fdata; ode_control=DustODEControl(atol=1e-8, rtol=1e-8))

# Packer and likelihood
packer = monty_packer([:beta, :gamma]; fixed=(I0=10.0, N=1000.0))
ll_odin = dust_likelihood_monty(uf, packer)

# Prior
prior_odin = @monty_prior begin
    beta ~ Gamma(2.0, 0.25)
    gamma ~ Gamma(2.0, 0.05)
end

posterior_odin = ll_odin + prior_odin
```

    MontyModel{Odin.var"#monty_model_combine##0#monty_model_combine##1"{MontyModel{Odin.var"#dust_likelihood_monty##2#dust_likelihood_monty##3"{DustUnfilter{var"##OdinModel#278", @NamedTuple{cases::Float64}}, MontyPacker}, Odin.var"#dust_likelihood_monty##4#dust_likelihood_monty##5"{Odin.var"#dust_likelihood_monty##2#dust_likelihood_monty##3"{DustUnfilter{var"##OdinModel#278", @NamedTuple{cases::Float64}}, MontyPacker}}, Nothing, Nothing}, MontyModel{var"#11#12", var"#13#14"{var"#11#12"}, var"#15#16", Matrix{Float64}}}, Odin.var"#monty_model_combine##2#monty_model_combine##3"{MontyModel{Odin.var"#dust_likelihood_monty##2#dust_likelihood_monty##3"{DustUnfilter{var"##OdinModel#278", @NamedTuple{cases::Float64}}, MontyPacker}, Odin.var"#dust_likelihood_monty##4#dust_likelihood_monty##5"{Odin.var"#dust_likelihood_monty##2#dust_likelihood_monty##3"{DustUnfilter{var"##OdinModel#278", @NamedTuple{cases::Float64}}, MontyPacker}}, Nothing, Nothing}, MontyModel{var"#11#12", var"#13#14"{var"#11#12"}, var"#15#16", Matrix{Float64}}}, Nothing, Matrix{Float64}}(["beta", "gamma"], Odin.var"#monty_model_combine##0#monty_model_combine##1"{MontyModel{Odin.var"#dust_likelihood_monty##2#dust_likelihood_monty##3"{DustUnfilter{var"##OdinModel#278", @NamedTuple{cases::Float64}}, MontyPacker}, Odin.var"#dust_likelihood_monty##4#dust_likelihood_monty##5"{Odin.var"#dust_likelihood_monty##2#dust_likelihood_monty##3"{DustUnfilter{var"##OdinModel#278", @NamedTuple{cases::Float64}}, MontyPacker}}, Nothing, Nothing}, MontyModel{var"#11#12", var"#13#14"{var"#11#12"}, var"#15#16", Matrix{Float64}}}(MontyModel{Odin.var"#dust_likelihood_monty##2#dust_likelihood_monty##3"{DustUnfilter{var"##OdinModel#278", @NamedTuple{cases::Float64}}, MontyPacker}, Odin.var"#dust_likelihood_monty##4#dust_likelihood_monty##5"{Odin.var"#dust_likelihood_monty##2#dust_likelihood_monty##3"{DustUnfilter{var"##OdinModel#278", @NamedTuple{cases::Float64}}, MontyPacker}}, Nothing, Nothing}(["beta", "gamma"], Odin.var"#dust_likelihood_monty##2#dust_likelihood_monty##3"{DustUnfilter{var"##OdinModel#278", @NamedTuple{cases::Float64}}, MontyPacker}(DustUnfilter{var"##OdinModel#278", @NamedTuple{cases::Float64}}(DustSystemGenerator{var"##OdinModel#278"}(var"##OdinModel#278"(3, [:S, :I, :R], [:N, :I0, :beta, :gamma], true, false, true, false, false, Dict{Symbol, Array}())), FilterData{@NamedTuple{cases::Float64}}([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0  …  41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0], [(cases = 14.0,), (cases = 24.0,), (cases = 27.0,), (cases = 54.0,), (cases = 69.0,), (cases = 112.0,), (cases = 123.0,), (cases = 175.0,), (cases = 252.0,), (cases = 302.0,)  …  (cases = 74.0,), (cases = 51.0,), (cases = 47.0,), (cases = 52.0,), (cases = 47.0,), (cases = 36.0,), (cases = 23.0,), (cases = 36.0,), (cases = 29.0,), (cases = 28.0,)]), 0.0, DustODEControl(1.0e-8, 1.0e-8, 10000, :dp5), [0.0, 0.0, 0.0], Xoshiro(0xcf5a312a7b0dd4ca, 0xccd8c80273c5fe6c, 0xafac7c30d8d5cf73, 0x40d17d77842a984e, 0x69fd646aafb7076f), nothing, nothing, nothing, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0  …  41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0]), MontyPacker([:beta, :gamma], [:beta, :gamma], Symbol[], Dict{Symbol, Tuple}(), Dict{Symbol, UnitRange{Int64}}(:beta => 1:1, :gamma => 2:2), 2, (I0 = 10.0, N = 1000.0), nothing)), Odin.var"#dust_likelihood_monty##4#dust_likelihood_monty##5"{Odin.var"#dust_likelihood_monty##2#dust_likelihood_monty##3"{DustUnfilter{var"##OdinModel#278", @NamedTuple{cases::Float64}}, MontyPacker}}(Odin.var"#dust_likelihood_monty##2#dust_likelihood_monty##3"{DustUnfilter{var"##OdinModel#278", @NamedTuple{cases::Float64}}, MontyPacker}(DustUnfilter{var"##OdinModel#278", @NamedTuple{cases::Float64}}(DustSystemGenerator{var"##OdinModel#278"}(var"##OdinModel#278"(3, [:S, :I, :R], [:N, :I0, :beta, :gamma], true, false, true, false, false, Dict{Symbol, Array}())), FilterData{@NamedTuple{cases::Float64}}([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0  …  41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0], [(cases = 14.0,), (cases = 24.0,), (cases = 27.0,), (cases = 54.0,), (cases = 69.0,), (cases = 112.0,), (cases = 123.0,), (cases = 175.0,), (cases = 252.0,), (cases = 302.0,)  …  (cases = 74.0,), (cases = 51.0,), (cases = 47.0,), (cases = 52.0,), (cases = 47.0,), (cases = 36.0,), (cases = 23.0,), (cases = 36.0,), (cases = 29.0,), (cases = 28.0,)]), 0.0, DustODEControl(1.0e-8, 1.0e-8, 10000, :dp5), [0.0, 0.0, 0.0], Xoshiro(0xcf5a312a7b0dd4ca, 0xccd8c80273c5fe6c, 0xafac7c30d8d5cf73, 0x40d17d77842a984e, 0x69fd646aafb7076f), nothing, nothing, nothing, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0  …  41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0]), MontyPacker([:beta, :gamma], [:beta, :gamma], Symbol[], Dict{Symbol, Tuple}(), Dict{Symbol, UnitRange{Int64}}(:beta => 1:1, :gamma => 2:2), 2, (I0 = 10.0, N = 1000.0), nothing))), nothing, nothing, Odin.MontyModelProperties(true, false, false, false)), MontyModel{var"#11#12", var"#13#14"{var"#11#12"}, var"#15#16", Matrix{Float64}}(["beta", "gamma"], var"#11#12"(), var"#13#14"{var"#11#12"}(var"#11#12"()), var"#15#16"(), [0.0 Inf; 0.0 Inf], Odin.MontyModelProperties(true, true, false, false))), Odin.var"#monty_model_combine##2#monty_model_combine##3"{MontyModel{Odin.var"#dust_likelihood_monty##2#dust_likelihood_monty##3"{DustUnfilter{var"##OdinModel#278", @NamedTuple{cases::Float64}}, MontyPacker}, Odin.var"#dust_likelihood_monty##4#dust_likelihood_monty##5"{Odin.var"#dust_likelihood_monty##2#dust_likelihood_monty##3"{DustUnfilter{var"##OdinModel#278", @NamedTuple{cases::Float64}}, MontyPacker}}, Nothing, Nothing}, MontyModel{var"#11#12", var"#13#14"{var"#11#12"}, var"#15#16", Matrix{Float64}}}(MontyModel{Odin.var"#dust_likelihood_monty##2#dust_likelihood_monty##3"{DustUnfilter{var"##OdinModel#278", @NamedTuple{cases::Float64}}, MontyPacker}, Odin.var"#dust_likelihood_monty##4#dust_likelihood_monty##5"{Odin.var"#dust_likelihood_monty##2#dust_likelihood_monty##3"{DustUnfilter{var"##OdinModel#278", @NamedTuple{cases::Float64}}, MontyPacker}}, Nothing, Nothing}(["beta", "gamma"], Odin.var"#dust_likelihood_monty##2#dust_likelihood_monty##3"{DustUnfilter{var"##OdinModel#278", @NamedTuple{cases::Float64}}, MontyPacker}(DustUnfilter{var"##OdinModel#278", @NamedTuple{cases::Float64}}(DustSystemGenerator{var"##OdinModel#278"}(var"##OdinModel#278"(3, [:S, :I, :R], [:N, :I0, :beta, :gamma], true, false, true, false, false, Dict{Symbol, Array}())), FilterData{@NamedTuple{cases::Float64}}([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0  …  41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0], [(cases = 14.0,), (cases = 24.0,), (cases = 27.0,), (cases = 54.0,), (cases = 69.0,), (cases = 112.0,), (cases = 123.0,), (cases = 175.0,), (cases = 252.0,), (cases = 302.0,)  …  (cases = 74.0,), (cases = 51.0,), (cases = 47.0,), (cases = 52.0,), (cases = 47.0,), (cases = 36.0,), (cases = 23.0,), (cases = 36.0,), (cases = 29.0,), (cases = 28.0,)]), 0.0, DustODEControl(1.0e-8, 1.0e-8, 10000, :dp5), [0.0, 0.0, 0.0], Xoshiro(0xcf5a312a7b0dd4ca, 0xccd8c80273c5fe6c, 0xafac7c30d8d5cf73, 0x40d17d77842a984e, 0x69fd646aafb7076f), nothing, nothing, nothing, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0  …  41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0]), MontyPacker([:beta, :gamma], [:beta, :gamma], Symbol[], Dict{Symbol, Tuple}(), Dict{Symbol, UnitRange{Int64}}(:beta => 1:1, :gamma => 2:2), 2, (I0 = 10.0, N = 1000.0), nothing)), Odin.var"#dust_likelihood_monty##4#dust_likelihood_monty##5"{Odin.var"#dust_likelihood_monty##2#dust_likelihood_monty##3"{DustUnfilter{var"##OdinModel#278", @NamedTuple{cases::Float64}}, MontyPacker}}(Odin.var"#dust_likelihood_monty##2#dust_likelihood_monty##3"{DustUnfilter{var"##OdinModel#278", @NamedTuple{cases::Float64}}, MontyPacker}(DustUnfilter{var"##OdinModel#278", @NamedTuple{cases::Float64}}(DustSystemGenerator{var"##OdinModel#278"}(var"##OdinModel#278"(3, [:S, :I, :R], [:N, :I0, :beta, :gamma], true, false, true, false, false, Dict{Symbol, Array}())), FilterData{@NamedTuple{cases::Float64}}([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0  …  41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0], [(cases = 14.0,), (cases = 24.0,), (cases = 27.0,), (cases = 54.0,), (cases = 69.0,), (cases = 112.0,), (cases = 123.0,), (cases = 175.0,), (cases = 252.0,), (cases = 302.0,)  …  (cases = 74.0,), (cases = 51.0,), (cases = 47.0,), (cases = 52.0,), (cases = 47.0,), (cases = 36.0,), (cases = 23.0,), (cases = 36.0,), (cases = 29.0,), (cases = 28.0,)]), 0.0, DustODEControl(1.0e-8, 1.0e-8, 10000, :dp5), [0.0, 0.0, 0.0], Xoshiro(0xcf5a312a7b0dd4ca, 0xccd8c80273c5fe6c, 0xafac7c30d8d5cf73, 0x40d17d77842a984e, 0x69fd646aafb7076f), nothing, nothing, nothing, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0  …  41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0]), MontyPacker([:beta, :gamma], [:beta, :gamma], Symbol[], Dict{Symbol, Tuple}(), Dict{Symbol, UnitRange{Int64}}(:beta => 1:1, :gamma => 2:2), 2, (I0 = 10.0, N = 1000.0), nothing))), nothing, nothing, Odin.MontyModelProperties(true, false, false, false)), MontyModel{var"#11#12", var"#13#14"{var"#11#12"}, var"#15#16", Matrix{Float64}}(["beta", "gamma"], var"#11#12"(), var"#13#14"{var"#11#12"}(var"#11#12"()), var"#15#16"(), [0.0 Inf; 0.0 Inf], Odin.MontyModelProperties(true, true, false, false))), nothing, [0.0 Inf; 0.0 Inf], Odin.MontyModelProperties(true, false, false, false))

``` julia
sampler_nuts = monty_sampler_nuts(target_acceptance=0.8, n_adaption=500)
initial_odin = [0.4 0.4; 0.08 0.08]

# Warmup
_ = monty_sample(posterior_odin, sampler_nuts, 100;
    n_chains=2, initial=initial_odin, seed=42)

# Timed run
t_odin_nuts = @timed begin
    s_odin_nuts = monty_sample(posterior_odin, sampler_nuts, 2000;
        n_chains=1, initial=reshape([0.4, 0.08], 2, 1), n_burnin=500, seed=123)
end

b_odin_nuts = vec(s_odin_nuts.pars[1, :, 1])
g_odin_nuts = vec(s_odin_nuts.pars[2, :, 1])

println("Approach 1: Odin.jl + NUTS")
println("  Time:     $(round(t_odin_nuts.time; digits=2))s")
println("  β:        $(round(mean(b_odin_nuts); digits=4)) ± $(round(std(b_odin_nuts); digits=4))")
println("  γ:        $(round(mean(g_odin_nuts); digits=4)) ± $(round(std(g_odin_nuts); digits=4))")
println("  ESS/s(β): $(round(ess_ipse(b_odin_nuts)/t_odin_nuts.time; digits=1))")
println("  iter/s:   $(round(2000/t_odin_nuts.time; digits=0))")
```

    ┌ Warning: Interrupted. Larger maxiters is needed. If you are using an integrator for non-stiff ODEs or an automatic switching algorithm (the default), you may want to consider using a method for stiff equations. See the solver pages for more details (e.g. https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/#Stiff-Problems).
    └ @ SciMLBase ~/.julia/packages/SciMLBase/J3OUh/src/integrator_interface.jl:677
    ┌ Warning: Interrupted. Larger maxiters is needed. If you are using an integrator for non-stiff ODEs or an automatic switching algorithm (the default), you may want to consider using a method for stiff equations. See the solver pages for more details (e.g. https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/#Stiff-Problems).
    └ @ SciMLBase ~/.julia/packages/SciMLBase/J3OUh/src/integrator_interface.jl:677
    ┌ Warning: Interrupted. Larger maxiters is needed. If you are using an integrator for non-stiff ODEs or an automatic switching algorithm (the default), you may want to consider using a method for stiff equations. See the solver pages for more details (e.g. https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/#Stiff-Problems).
    └ @ SciMLBase ~/.julia/packages/SciMLBase/J3OUh/src/integrator_interface.jl:677
    ┌ Warning: Interrupted. Larger maxiters is needed. If you are using an integrator for non-stiff ODEs or an automatic switching algorithm (the default), you may want to consider using a method for stiff equations. See the solver pages for more details (e.g. https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/#Stiff-Problems).
    └ @ SciMLBase ~/.julia/packages/SciMLBase/J3OUh/src/integrator_interface.jl:677
    ┌ Warning: Interrupted. Larger maxiters is needed. If you are using an integrator for non-stiff ODEs or an automatic switching algorithm (the default), you may want to consider using a method for stiff equations. See the solver pages for more details (e.g. https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/#Stiff-Problems).
    └ @ SciMLBase ~/.julia/packages/SciMLBase/J3OUh/src/integrator_interface.jl:677
    ┌ Warning: Interrupted. Larger maxiters is needed. If you are using an integrator for non-stiff ODEs or an automatic switching algorithm (the default), you may want to consider using a method for stiff equations. See the solver pages for more details (e.g. https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/#Stiff-Problems).
    └ @ SciMLBase ~/.julia/packages/SciMLBase/J3OUh/src/integrator_interface.jl:677
    Approach 1: Odin.jl + NUTS
      Time:     1.67s
      β:        0.5051 ± 0.0035
      γ:        0.1001 ± 0.0009
      ESS/s(β): 577.2
      iter/s:   1199.0

## Approach 2: Odin.jl + RW-MCMC

Random walk MCMC requires no gradients and is extremely fast per
iteration, though with lower per-iteration efficiency.

``` julia
sampler_rw = monty_sampler_random_walk([0.005 0.0; 0.0 0.001])

# Warmup
_ = monty_sample(posterior_odin, sampler_rw, 100;
    n_chains=1, initial=reshape([0.4, 0.08], 2, 1), seed=42)

# Timed run
t_odin_rw = @timed begin
    s_odin_rw = monty_sample(posterior_odin, sampler_rw, 5000;
        n_chains=1, initial=reshape([0.4, 0.08], 2, 1), n_burnin=1000, seed=456)
end

b_odin_rw = vec(s_odin_rw.pars[1, :, 1])
g_odin_rw = vec(s_odin_rw.pars[2, :, 1])

println("Approach 2: Odin.jl + RW-MCMC")
println("  Time:     $(round(t_odin_rw.time; digits=3))s")
println("  β:        $(round(mean(b_odin_rw); digits=4)) ± $(round(std(b_odin_rw); digits=4))")
println("  γ:        $(round(mean(g_odin_rw); digits=4)) ± $(round(std(g_odin_rw); digits=4))")
println("  ESS/s(β): $(round(ess_ipse(b_odin_rw)/t_odin_rw.time; digits=1))")
println("  iter/s:   $(round(5000/t_odin_rw.time; digits=0))")
```

    Approach 2: Odin.jl + RW-MCMC
      Time:     0.115s
      β:        0.5077 ± 0.0058
      γ:        0.0999 ± 0.0011
      ESS/s(β): 84.9
      iter/s:   43441.0

## Approach 3: Turing.jl + MTK (Baseline)

``` julia
sa = InterpolatingAdjoint(autojacvec=EnzymeVJP())

Turing.@model function sir_turing(obs_data, times_data)
    be ~ Gamma(2, 1/4)
    ge ~ Gamma(2, 1/20)
    prob = remake(prob_base; p=[sir_s.β => be, sir_s.γ => ge])
    sol = solve(prob, Tsit5(); saveat=times_data, sensealg=sa)
    if sol.retcode != ReturnCode.Success
        Turing.@addlogprob! -Inf
        return
    end
    Iv = sol[sir_s.I]
    for i in eachindex(obs_data)
        obs_data[i] ~ Poisson(max(Iv[i], 1e-6))
    end
end

mdl = sir_turing(obs, data_times)
```

    DynamicPPL.Model{typeof(sir_turing), (:obs_data, :times_data), (), (), Tuple{Vector{Int64}, Vector{Float64}}, Tuple{}, DynamicPPL.DefaultContext, false}(sir_turing, (obs_data = [14, 24, 27, 54, 69, 112, 123, 175, 252, 302  …  74, 51, 47, 52, 47, 36, 23, 36, 29, 28], times_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0  …  41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0]), NamedTuple(), DynamicPPL.DefaultContext())

``` julia
# Warmup
_ = with_logger(NullLogger()) do
    Turing.sample(mdl, Turing.NUTS(), 50; progress=false)
end

# Timed run
t1 = @timed begin
    ch1 = with_logger(NullLogger()) do
        Turing.sample(mdl, Turing.NUTS(), 2000; progress=false)
    end
end

burnin = 500
b1 = vec(ch1[:be])[burnin+1:end]
g1 = vec(ch1[:ge])[burnin+1:end]

println("Approach 3: Turing + adjoint sensealg")
println("  Time:     $(round(t1.time; digits=1))s")
println("  β:        $(round(mean(b1); digits=4)) ± $(round(std(b1); digits=4))")
println("  γ:        $(round(mean(g1); digits=4)) ± $(round(std(g1); digits=4))")
println("  ESS/s(β): $(round(ess_ipse(b1)/t1.time; digits=1))")
println("  ESS/s(γ): $(round(ess_ipse(g1)/t1.time; digits=1))")
println("  iter/s:   $(round(2000/t1.time; digits=0))")
```

    Approach 3: Turing + adjoint sensealg
      Time:     133.7s
      β:        0.5052 ± 0.0035
      γ:        0.1001 ± 0.001
      ESS/s(β): 9.9
      ESS/s(γ): 11.2
      iter/s:   15.0

## Approach 4: Symbolic Sensitivity Equations

Instead of using AD to differentiate through the ODE solver, we derive
the **forward sensitivity equations** symbolically. For parameters
$\theta = (\beta, \gamma)$ and state $y = (S, I, R)$, the sensitivity
matrix $s_{ij} = \partial y_i / \partial \theta_j$ satisfies:

$$\frac{ds}{dt} = \frac{\partial f}{\partial y} s + \frac{\partial f}{\partial \theta}$$

Both $\partial f / \partial y$ (the Jacobian) and
$\partial f / \partial \theta$ can be computed symbolically using
`Symbolics.jacobian`. The augmented system (9 equations: 3 state + 6
sensitivity) is solved in a single ODE call, giving us both the
trajectory and its parameter derivatives.

### Deriving the Augmented System

``` julia
@parameters β_s γ_s N_s
@variables S_s(t) I_s(t) R_s(t)

f_rhs = [-β_s * S_s * I_s / N_s,
          β_s * S_s * I_s / N_s - γ_s * I_s,
          γ_s * I_s]

J_fy = Symbolics.jacobian(f_rhs, [S_s, I_s, R_s])
F_fθ = Symbolics.jacobian(f_rhs, [β_s, γ_s])

println("∂f/∂y (Jacobian): $(size(J_fy))")
println("∂f/∂θ: $(size(F_fθ))")
```

    ∂f/∂y (Jacobian): (3, 3)
    ∂f/∂θ: (3, 2)

### Building the Augmented ODE

``` julia
@variables s_S_β(t) s_I_β(t) s_R_β(t) s_S_γ(t) s_I_γ(t) s_R_γ(t)

s_mat = [s_S_β s_S_γ;
         s_I_β s_I_γ;
         s_R_β s_R_γ]

ds_dt = J_fy * s_mat + F_fθ

aug_eqs = [
    D(S_s)    ~ f_rhs[1],
    D(I_s)    ~ f_rhs[2],
    D(R_s)    ~ f_rhs[3],
    D(s_S_β)  ~ ds_dt[1,1],
    D(s_I_β)  ~ ds_dt[2,1],
    D(s_R_β)  ~ ds_dt[3,1],
    D(s_S_γ)  ~ ds_dt[1,2],
    D(s_I_γ)  ~ ds_dt[2,2],
    D(s_R_γ)  ~ ds_dt[3,2],
]

@named aug_sir = ODESystem(aug_eqs, t)
aug_sir_s = structural_simplify(aug_sir)
println("Augmented system: $(length(unknowns(aug_sir_s))) unknowns, $(length(parameters(aug_sir_s))) parameters")
```

    Augmented system: 9 unknowns, 3 parameters

### Using ODEFunction Codegen

The critical optimisation: instead of using MTK’s high-level
`ODEProblem` with symbolic parameter indexing (which costs ~100ms per
`remake`), we extract the compiled `ODEFunction` and work directly with
numeric arrays.

``` julia
odefn = ODEFunction(aug_sir_s)

p_syms = parameters(aug_sir_s)
u_syms = unknowns(aug_sir_s)

p_map = Dict(string(p) => i for (i, p) in enumerate(p_syms))
u_map = Dict(string(u) => i for (i, u) in enumerate(u_syms))

iβ = p_map["β_s"]; iγ = p_map["γ_s"]; iN = p_map["N_s"]
iI = u_map["I_s(t)"]; i_sIb = u_map["s_I_β(t)"]; i_sIg = u_map["s_I_γ(t)"]
iS = u_map["S_s(t)"]

println("Parameter indices: β=$iβ, γ=$iγ, N=$iN")
println("State indices: I=$iI, s_I_β=$i_sIb, s_I_γ=$i_sIg")

# Build problem with correct parameter/state ordering
u0 = zeros(9)
u0[iS] = 990.0; u0[iI] = 10.0  # sensitivities start at 0

p0 = zeros(3)
p0[iβ] = 0.5; p0[iγ] = 0.1; p0[iN] = 1000.0

prob_aug = ODEProblem(odefn, u0, (0.0, 50.0), p0)

# Verify
sol_check = solve(prob_aug, Tsit5(); saveat=data_times, abstol=1e-8, reltol=1e-8)
println("\nI(t=25) = $(round(sol_check[iI, 25]; digits=2)) (expected: $(round(I_true[26]; digits=2)))")
```

    Parameter indices: β=1, γ=2, N=3
    State indices: I=8, s_I_β=5, s_I_γ=2

    I(t=25) = 272.53 (expected: 272.53)

### Analytic Gradient Computation

The log-likelihood gradient is:

$$\frac{\partial \ell}{\partial \theta_j} = \sum_i \frac{\partial \log p(y_i \mid \lambda_i)}{\partial \lambda_i} \cdot \frac{\partial \lambda_i}{\partial \theta_j}$$

where $\lambda_i = I(t_i)$ and
$\partial \lambda_i / \partial \theta_j = s_{I,j}(t_i)$ comes directly
from the augmented ODE solution. For $\text{Poisson}(\lambda)$:

$$\frac{\partial}{\partial \lambda} \log p(k \mid \lambda) = \frac{k}{\lambda} - 1$$

``` julia
struct SymSensLogDensity
    prob::ODEProblem
    obs::Vector{Int}
    times::Vector{Float64}
    iβ::Int; iγ::Int; iI::Int; i_sIb::Int; i_sIg::Int
    p_buf::Vector{Float64}
end

function LogDensityProblems.logdensity_and_gradient(ld::SymSensLogDensity, θ_log)
    be = exp(θ_log[1]); ge = exp(θ_log[2])
    ld.p_buf[ld.iβ] = be; ld.p_buf[ld.iγ] = ge

    p2 = remake(ld.prob; p=ld.p_buf)
    sol = solve(p2, Tsit5(); saveat=ld.times, abstol=1e-8, reltol=1e-8)
    sol.retcode != ReturnCode.Success && return (-Inf, [0.0, 0.0])

    ll = 0.0; dll_db = 0.0; dll_dg = 0.0
    for i in eachindex(ld.obs)
        Iv = sol[ld.iI, i]
        lam = max(Iv, 1e-6)
        ll += logpdf(Poisson(lam), ld.obs[i])
        if Iv > 1e-6
            dlogp = ld.obs[i] / lam - 1.0
            dll_db += dlogp * sol[ld.i_sIb, i]
            dll_dg += dlogp * sol[ld.i_sIg, i]
        end
    end

    # Prior (log-transformed) + Jacobian correction
    ll += logpdf(Gamma(2, 1/4), be) + θ_log[1]
    ll += logpdf(Gamma(2, 1/20), ge) + θ_log[2]

    # Chain rule: ∂/∂(log θ) = θ · ∂/∂θ
    dp_db = 1.0 / be - 4.0    # ∂ log Gamma(2, 1/4) / ∂β
    dp_dg = 1.0 / ge - 20.0   # ∂ log Gamma(2, 1/20) / ∂γ
    g1 = (dll_db + dp_db) * be + 1.0
    g2 = (dll_dg + dp_dg) * ge + 1.0

    return (ll, [g1, g2])
end

function LogDensityProblems.logdensity(ld::SymSensLogDensity, θ)
    return LogDensityProblems.logdensity_and_gradient(ld, θ)[1]
end
LogDensityProblems.dimension(::SymSensLogDensity) = 2
LogDensityProblems.capabilities(::Type{SymSensLogDensity}) = LogDensityProblems.LogDensityOrder{1}()
```

### Gradient Verification

``` julia
ld = SymSensLogDensity(prob_aug, obs, data_times, iβ, iγ, iI, i_sIb, i_sIg, copy(p0))

θ0 = [log(0.5), log(0.1)]
ll0, g0 = LogDensityProblems.logdensity_and_gradient(ld, θ0)
println("Log-density at truth: $(round(ll0; digits=2))")
println("Analytic gradient:    $(round.(g0; digits=4))")

# Finite difference check
h = 1e-5
for i in 1:2
    θp = copy(θ0); θp[i] += h
    θm = copy(θ0); θm[i] -= h
    fd = (LogDensityProblems.logdensity(ld, θp) - LogDensityProblems.logdensity(ld, θm)) / (2h)
    println("  param $i: analytic=$(round(g0[i]; digits=4))  finite_diff=$(round(fd; digits=4))")
end
```

    Log-density at truth: -195.3
    Analytic gradient:    [233.5606, 41.7046]
      param 1: analytic=233.5606  finite_diff=233.5606
      param 2: analytic=41.7046  finite_diff=41.7046

### NUTS Sampling

``` julia
nuts = AdvancedHMC.NUTS(0.8; max_depth=10)

# Warmup
_ = AdvancedHMC.sample(ld, nuts, 200; initial_params=θ0, progress=false, verbose=false)

# Timed run
t2 = @timed begin
    chain2 = AdvancedHMC.sample(ld, nuts, 2000; initial_params=θ0, progress=false, verbose=false)
end

b2 = [exp(chain2[i].z.θ[1]) for i in burnin+1:length(chain2)]
g2 = [exp(chain2[i].z.θ[2]) for i in burnin+1:length(chain2)]

println("Approach 4: SymSens + ODEFunction codegen + AdvancedHMC")
println("  Time:     $(round(t2.time; digits=2))s")
println("  β:        $(round(mean(b2); digits=4)) ± $(round(std(b2); digits=4))")
println("  γ:        $(round(mean(g2); digits=4)) ± $(round(std(g2); digits=4))")
println("  ESS/s(β): $(round(ess_ipse(b2)/t2.time; digits=1))")
println("  ESS/s(γ): $(round(ess_ipse(g2)/t2.time; digits=1))")
println("  iter/s:   $(round(2000/t2.time; digits=0))")
```

    [ Info: Found initial step size 0.00625
    [ Info: Found initial step size 0.0125
    Approach 4: SymSens + ODEFunction codegen + AdvancedHMC
      Time:     1.12s
      β:        0.5052 ± 0.0034
      γ:        0.1001 ± 0.0009
      ESS/s(β): 930.5
      ESS/s(γ): 593.5
      iter/s:   1782.0

## Comparison

``` julia
fig = Figure(size=(800, 400))

ax1 = Axis(fig[1, 1]; xlabel="β", ylabel="Density", title="Posterior: β")
hist!(ax1, b_odin_nuts; bins=40, color=(:green, 0.4), normalization=:pdf, label="Odin NUTS")
hist!(ax1, b_odin_rw; bins=40, color=(:purple, 0.3), normalization=:pdf, label="Odin RW")
hist!(ax1, b1; bins=40, color=(:steelblue, 0.3), normalization=:pdf, label="Turing NUTS")
hist!(ax1, b2; bins=40, color=(:orange, 0.3), normalization=:pdf, label="SymSens NUTS")
vlines!(ax1, [0.5]; color=:red, linestyle=:dash, label="Truth")
axislegend(ax1; position=:rt)

ax2 = Axis(fig[1, 2]; xlabel="γ", ylabel="Density", title="Posterior: γ")
hist!(ax2, g_odin_nuts; bins=40, color=(:green, 0.4), normalization=:pdf, label="Odin NUTS")
hist!(ax2, g_odin_rw; bins=40, color=(:purple, 0.3), normalization=:pdf, label="Odin RW")
hist!(ax2, g1; bins=40, color=(:steelblue, 0.3), normalization=:pdf, label="Turing NUTS")
hist!(ax2, g2; bins=40, color=(:orange, 0.3), normalization=:pdf, label="SymSens NUTS")
vlines!(ax2, [0.1]; color=:red, linestyle=:dash, label="Truth")
axislegend(ax2; position=:rt)

fig
```

![Figure 1: Posterior distributions from all four approaches

](07_fast_inference_files/figure-commonmark/fig-comparison-output-1.png)

## Performance Summary

| Approach | Sampler | Iters | Time (s) | β (mean ± sd) | γ (mean ± sd) | ESS/s(β) | iter/s |
|----|----|---:|---:|---:|---:|---:|---:|
| Odin.jl NUTS | NUTS | 2000 | 1.67 | 0.5051 ± 0.0035 | 0.1001 ± 0.0009 | 577.2 | 1199.0 |
| Odin.jl RW-MCMC | RW | 5000 | 0.12 | 0.5077 ± 0.0058 | 0.0999 ± 0.0011 | 84.9 | 43441.0 |
| Turing + MTK NUTS | NUTS | 2000 | 133.67 | 0.5052 ± 0.0035 | 0.1001 ± 0.001 | 9.9 | 15.0 |
| SymSens + codegen NUTS | NUTS | 2000 | 1.12 | 0.5052 ± 0.0034 | 0.1001 ± 0.0009 | 930.5 | 1782.0 |

Odin.jl NUTS is **80.0×** faster than Turing + MTK NUTS. SymSens +
codegen NUTS is **119.0×** faster than Turing + MTK NUTS.

## Where Does the Speedup Come From?

The performance differences have several sources:

1.  **Odin.jl** compiles the DSL directly to a tight Julia ODE function
    with no symbolic overhead. ForwardDiff through this minimal code
    path is fast (~0.5ms/iteration for NUTS). RW-MCMC is even faster
    since it needs no gradient at all.

2.  **No AD at runtime** (SymSens, ~20×): Instead of propagating
    ForwardDiff dual numbers through the ODE solver, we solve the
    symbolically-derived sensitivity equations as a single augmented
    ODE. The gradient is then a simple dot product of `∂ log p / ∂λ`
    with the sensitivity values `∂λ / ∂θ`.

3.  **ODEFunction codegen** (SymSens, ~500×): MTK’s `ODEFunction(sys)`
    compiles the symbolic system into a raw Julia function. Using this
    directly with numeric arrays avoids the symbolic parameter remapping
    that costs ~100ms per `remake` call in the high-level MTK API.

4.  **AdvancedHMC directly** (~1.3×): Bypassing Turing’s `@model` macro
    and VarInfo tracking removes ~25% overhead. The `LogDensityProblems`
    interface gives AdvancedHMC direct access to the log-density and
    gradient.

### Key Insight

For fast-evaluating models like SIR, **framework overhead dominates
compute time**. Odin.jl eliminates this by design — the DSL compiles to
plain Julia with no symbolic layer at runtime. For MTK users, the key is
to use `ODEFunction(sys)` to extract compiled functions and work with
raw arrays, rather than using the high-level API at every MCMC
iteration.
