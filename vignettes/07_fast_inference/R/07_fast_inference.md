# Bayesian Inference with odin2/dust2/monty


- [Overview](#overview)
- [Model and Data](#model-and-data)
- [ESS Helper](#ess-helper)
- [Inference: Random-Walk MCMC](#inference-random-walk-mcmc)
- [Results](#results)
- [Notes](#notes)

## Overview

This R companion demonstrates Bayesian inference for the SIR model using
odin2/dust2/monty. Since R’s odin2 does not currently expose adjoint
gradients through its public API, only random-walk MCMC is available
(not HMC/NUTS).

``` r
library(odin2)
library(dust2)
library(monty)
```

## Model and Data

``` r
sir <- odin({
  deriv(S) <- -beta * S * I / N
  deriv(I) <- beta * S * I / N - gamma * I
  deriv(R) <- gamma * I
  initial(S) <- N - I0
  initial(I) <- I0
  initial(R) <- 0
  beta <- parameter(0.5)
  gamma <- parameter(0.1)
  I0 <- parameter(10)
  N <- parameter(1000)
  obs <- data()
  obs ~ Poisson(max(I, 1e-6))
})
```

    ✔ Wrote 'DESCRIPTION'

    ✔ Wrote 'NAMESPACE'

    ✔ Wrote 'R/dust.R'

    ✔ Wrote 'src/dust.cpp'

    ✔ Wrote 'src/Makevars'

    ℹ 28 functions decorated with [[cpp11::register]]

    ✔ generated file 'cpp11.R'

    ✔ generated file 'cpp11.cpp'

    ℹ Re-compiling odin.system11ab63df

    ── R CMD INSTALL ───────────────────────────────────────────────────────────────
    * installing *source* package ‘odin.system11ab63df’ ...
    ** this is package ‘odin.system11ab63df’ version ‘0.0.1’
    ** using staged installation
    ** libs
    using C++ compiler: ‘Apple clang version 17.0.0 (clang-1700.0.13.5)’
    using SDK: ‘MacOSX15.5.sdk’
    /usr/bin/clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG  -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/cpp11/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/dust2/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/monty/include' -I/opt/R/arm64/include   -DHAVE_INLINE   -fPIC  -falign-functions=64 -Wall -g -O2  -Wall -pedantic  -c cpp11.cpp -o cpp11.o
    /usr/bin/clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG  -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/cpp11/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/dust2/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/monty/include' -I/opt/R/arm64/include   -DHAVE_INLINE   -fPIC  -falign-functions=64 -Wall -g -O2  -Wall -pedantic  -c dust.cpp -o dust.o
    /usr/bin/clang++ -arch arm64 -std=gnu++17 -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -L/Library/Frameworks/R.framework/Resources/lib -L/opt/R/arm64/lib -o odin.system11ab63df.so cpp11.o dust.o -F/Library/Frameworks/R.framework/.. -framework R
    installing to /private/var/folders/yh/30rj513j6mn1n7x556c2v4w80000gn/T/Rtmp5gRb1W/devtools_install_9d825991ea93/00LOCK-dust_9d82415a7ed1/00new/odin.system11ab63df/libs
    ** checking absolute paths in shared objects and dynamic libraries
    * DONE (odin.system11ab63df)

    ℹ Loading odin.system11ab63df

``` r
pars <- list(beta = 0.5, gamma = 0.1, I0 = 10, N = 1000)
sys <- dust_system_create(sir, pars,
  ode_control = dust_ode_control(atol = 1e-8, rtol = 1e-8))
dust_system_set_state_initial(sys)
times <- 1:50
state <- dust_system_simulate(sys, times)

set.seed(1)
I_vals <- state[2, ]
obs <- rpois(length(times), pmax(I_vals, 1e-6))
data <- data.frame(time = times, obs = obs)

cat("Generated", length(obs), "observations, range:",
    range(obs), "\n")
```

    Generated 50 observations, range: 12 500 

## ESS Helper

``` r
ess_ipse <- function(x) {
  n <- length(x)
  mu <- mean(x)
  var0 <- sum((x - mu)^2) / n
  if (var0 == 0) return(n)
  max_lag <- min(n - 1, 1000)
  rho <- sapply(1:max_lag, function(k) {
    sum((x[1:(n - k)] - mu) * (x[(k + 1):n] - mu)) / (n * var0)
  })
  tau <- 1
  for (k in seq(1, max_lag - 1, by = 2)) {
    ps <- rho[k] + rho[k + 1]
    if (ps < 0) break
    tau <- tau + 2 * ps
  }
  n / tau
}
```

## Inference: Random-Walk MCMC

R odin2 does not currently provide gradients via its public API, so
HMC/NUTS sampling is not available. We use random-walk
Metropolis–Hastings.

``` r
d <- dust_unfilter_create(sir, data = data, time_start = 0,
  ode_control = dust_ode_control(atol = 1e-8, rtol = 1e-8))
packer <- monty_packer(c("beta", "gamma"),
  fixed = list(I0 = 10, N = 1000))
ll <- dust_likelihood_monty(d, packer)

prior <- monty_dsl({
  beta ~ Gamma(shape = 2, rate = 4)
  gamma ~ Gamma(shape = 2, rate = 20)
})
posterior <- ll + prior
```

``` r
vcv <- diag(c(0.0001, 0.000004))
sampler <- monty_sampler_random_walk(vcv)
```

``` r
# Warmup
invisible(monty_sample(posterior, sampler, 200,
  initial = matrix(c(0.5, 0.1), ncol = 1)))
```

    ⡀⠀ Sampling  ■                                |   0% ETA:  1s

    ✔ Sampled 200 steps across 1 chain in 40ms

``` r
# Timed run
t0 <- proc.time()
samples <- monty_sample(posterior, sampler, 5000,
  initial = matrix(c(0.5, 0.1), ncol = 1))
elapsed <- (proc.time() - t0)[3]

burnin <- 1000
bs <- samples$pars[1, (burnin + 1):5000, 1]
gs <- samples$pars[2, (burnin + 1):5000, 1]
```

## Results

``` r
cat(sprintf("R odin2 Random-Walk MCMC\n"))
```

    R odin2 Random-Walk MCMC

``` r
cat(sprintf("  Time:     %.2fs\n", elapsed))
```

      Time:     0.43s

``` r
cat(sprintf("  β:        %.4f ± %.4f\n", mean(bs), sd(bs)))
```

      β:        0.5003 ± 0.0033

``` r
cat(sprintf("  γ:        %.4f ± %.4f\n", mean(gs), sd(gs)))
```

      γ:        0.0999 ± 0.0009

``` r
cat(sprintf("  ESS(β):   %.1f\n", ess_ipse(bs)))
```

      ESS(β):   560.2

``` r
cat(sprintf("  ESS(γ):   %.1f\n", ess_ipse(gs)))
```

      ESS(γ):   375.0

``` r
cat(sprintf("  ESS/s(β): %.1f\n", ess_ipse(bs) / elapsed))
```

      ESS/s(β): 1308.8

``` r
cat(sprintf("  ESS/s(γ): %.1f\n", ess_ipse(gs) / elapsed))
```

      ESS/s(γ): 876.2

``` r
cat(sprintf("  iter/s:   %.0f\n", 5000 / elapsed))
```

      iter/s:   11682

``` r
par(mfrow = c(1, 2))

hist(bs, breaks = 40, col = "steelblue", border = "white",
     main = expression("Posterior: " * beta),
     xlab = expression(beta), probability = TRUE)
abline(v = 0.5, col = "red", lty = 2, lwd = 2)

hist(gs, breaks = 40, col = "steelblue", border = "white",
     main = expression("Posterior: " * gamma),
     xlab = expression(gamma), probability = TRUE)
abline(v = 0.1, col = "red", lty = 2, lwd = 2)
```

<div id="fig-posteriors">

![](07_fast_inference_files/figure-commonmark/fig-posteriors-1.png)

Figure 1: Posterior distributions (R odin2 RW-MCMC)

</div>

## Notes

The R odin2/dust2/monty stack uses compiled C++ for the ODE solver and
likelihood evaluation, making random-walk MCMC very fast (~12,500
iter/s). However, without gradient information, HMC/NUTS cannot be used.

The Julia MTK approach in vignette 07 demonstrates how symbolic
sensitivity equations can provide exact gradients, enabling NUTS
sampling with ~1,200 iter/s and much higher effective sample sizes per
iteration.
