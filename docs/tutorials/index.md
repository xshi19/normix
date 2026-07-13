# Tutorials

Executable, end-to-end tutorials covering every public feature of normix. Each
page runs against the current source at build time, so the code and outputs you
see are always in sync with the library.

The tutorials are grouped into five tracks. **Core** builds the conceptual
foundation; **Distributions** tours every class; **EM** covers fitting in
practice; **Statistics** adds model comparison and goodness of fit; and
**Finance** applies everything to real market data.

Formal derivations live in the {doc}`../theory/index` — tutorials pull in the
key formulas where they help, and link out for the full math. Some theory
topics (shrinkage, ENB) are theory-only until matching
tutorials land.

```{toctree}
:maxdepth: 1
:caption: Core concepts

core/01_exponential_family
core/02_gh_family_tour
core/03_bessel_and_log_kv
core/04_random_sampling
```

```{toctree}
:maxdepth: 1
:caption: Distribution tour

distributions/01_univariate_positive
distributions/02_gig
distributions/03_multivariate_normal
distributions/04_normal_mixtures
distributions/05_factor_mixtures
```

```{toctree}
:maxdepth: 1
:caption: EM in practice

em/01_batch_em
em/02_incremental_em
em/03_initialization_and_multistart
em/04_em_vs_mcecm
```

```{toctree}
:maxdepth: 1
:caption: Statistical analysis

stats/01_divergences
stats/02_goodness_of_fit
stats/03_varentropy
```

```{toctree}
:maxdepth: 1
:caption: Finance

finance/01_univariate_index
finance/02_multivariate_stocks
finance/03_factor_mixture_portfolios
finance/04_cvar_optimization
finance/05_mean_risk_optimization
finance/06_transaction_costs
```
