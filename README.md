# odin-mtk-vignettes

Epidemiological modelling vignettes ported from the R [odin2](https://github.com/mrc-ide/odin2)/[dust2](https://github.com/mrc-ide/dust2)/[monty](https://github.com/mrc-ide/monty) ecosystem to Julia using [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/), [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/), and [Turing.jl](https://turinglang.org/).

## Motivation

The R odin2/dust2/monty packages provide a powerful DSL-based workflow for infectious disease modelling:

- **odin2**: Model definition DSL (ODEs, discrete-time stochastic)
- **dust2**: Simulation engine (particle filters, ODE solvers)
- **monty**: Bayesian inference (MCMC samplers, priors)

The Julia SciML ecosystem offers equivalent (and in many cases more flexible) functionality via standard packages:

| R (odin2/dust2/monty) | Julia equivalent |
|---|---|
| `odin({ deriv(S) <- ... })` | `@mtkmodel` / `ODESystem` (ModelingToolkit.jl) |
| `dust_system_create()` / `dust_system_simulate()` | `ODEProblem` / `solve()` (DifferentialEquations.jl) |
| `update()` with `Binomial()` | `Distributions.jl` + discrete-time loop |
| `dust_filter_create()` / `dust_likelihood_run()` | Custom bootstrap particle filter |
| `monty_dsl({ beta ~ Gamma(...) })` | `@model` (Turing.jl) |
| `monty_sampler_random_walk()` / `monty_sample()` | `sample(model, NUTS(), ...)` (Turing.jl) |

## Vignettes

Each vignette has a Julia version (using MTK/Turing) and an R companion (using odin2/dust2/monty) in its `R/` subdirectory.

| # | Title | Description |
|--:|-------|-------------|
| 01 | [Basic ODE Model: SIR](vignettes/01_basic_ode/01_basic_ode.md) | Deterministic SIR with `@mtkmodel` and `ODEProblem` |
| 02 | [Stochastic Discrete-Time SIR](vignettes/02_stochastic/02_stochastic.md) | Binomial stochastic model with `Distributions.jl` |
| 03 | [Incidence Tracking](vignettes/03_observations/03_observations.md) | Daily incidence counters (equivalent to `zero_every`) |
| 04 | [Age-Structured SIR](vignettes/04_arrays/04_arrays.md) | Vector state variables in ModelingToolkit.jl |
| 05 | [Particle Filter and Likelihood](vignettes/05_particle_filter/05_particle_filter.md) | Bootstrap particle filter from scratch |
| 06 | [Bayesian Inference with MCMC](vignettes/06_inference/06_inference.md) | ODE + Turing.jl NUTS for full posterior inference |
| 07 | [Fast Inference with Symbolic Sensitivities](vignettes/07_fast_inference/07_fast_inference.md) | ~110× speedup via symbolic sensitivity equations + codegen |

## Getting Started

### Prerequisites

- [Julia](https://julialang.org/) ≥ 1.10
- [Quarto](https://quarto.org/) (for rendering vignettes)
- R with `odin2`, `dust2`, `monty` (for R companion vignettes)

### Rendering a vignette

```bash
cd vignettes
quarto render 01_basic_ode/01_basic_ode.qmd     # Julia version
quarto render 01_basic_ode/R/01_basic_ode.qmd    # R companion
```

### Rendering all vignettes

```bash
cd vignettes
quarto render
```

## Structure

```
vignettes/
├── _quarto.yml              # Quarto project config
├── Project.toml              # Julia dependencies
├── 01_basic_ode/
│   ├── 01_basic_ode.qmd     # Julia (MTK/Turing)
│   └── R/
│       └── 01_basic_ode.qmd # R (odin2/dust2/monty)
├── 02_stochastic/
│   ├── ...
...
```

## See Also

- [Odin.jl](https://github.com/epirecipes/Odin.jl) — Direct Julia port of odin2/dust2/monty with a custom DSL
- [epirecipes](https://github.com/epirecipes) — Epidemiological modelling recipes

## License

MIT

