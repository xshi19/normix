# Benchmark trends

Per-commit timing for a small [ASV](https://asv.readthedocs.io/) suite: Bessel
`log_kv`, GIG `from_expectation`, the EM E-step, a short EM fit, compile time,
and GIG sampling. CPU and CUDA are separate series.

```{raw} html
<p><a href="benchmarks/">Open the ASV dashboard</a></p>
```

The JSON for those plots lives in `asv_bench/results/`, recorded on the
maintainer's Linux desktop (GPU) at each GitHub release. Docs CI rebuilds the
dashboard from that JSON; it does not re-time the suite. Pull requests run a
CPU-only Bessel / GIG `asv continuous` check (factor 1.5) on GitHub-hosted
runners — a smoke tripwire, not the trend series.

Investigation scripts (S&P 500, MCECM, large sweeps) stay in
[`benchmarks/`](https://github.com/xshi19/normix/tree/master/benchmarks)
on GitHub.
