# Research notes

Working notes that are not part of the library contract. They use the
public API only; nothing here changes `normix`.

The first note asks whether a *portfolio* can reveal the latent
subordinator $Y$ in a normal mean–variance mixture, and then measures
that claim on daily S&P 500 returns.

```{toctree}
:maxdepth: 1

subordinator_tracking
subordinator_tracking_empirics
```

## Subordinator tracking — one-page summary

In the mixture $X \stackrel{d}{=} \mu + \gamma Y + \sqrt{Y}\,Z$, the
linear functional of $X$ with the least Gaussian noise per unit of
$Y$-loading is unique: $w^\star \propto \Sigma^{-1}\gamma$. The
achievable signal-to-noise is the scalar
$\tilde q = \gamma^\top\Sigma^{-1}\gamma$ that already appears in the
GH density and the GIG posterior. That direction is also the best
linear predictor of $Y$, the model's max-skewness portfolio (for VG,
NIG, and NInvG; for GH under a cumulant check), and the Markowitz
tangency fund when the whole risk premium is compensation for $Y$.

A second, non-tradable channel learns $Y$ from the orthogonal
Mahalanobis radius $q_\perp(x)$. It does not need $\gamma$, and it
gets stronger with dimension.

On the S&P 500 current-constituent panel (2015–2026, $d\le 468$) the
algebra is not contradicted — the MSE laws hold in simulation, and
every fitted VG/NIG/GH model has the tracker as its max-skewness
direction — but the linear channel is absent. Fitted fluctuation SNR
sits at or below a day-wise sign-flip null at every $d$. The posterior
mean $E[Y\mid X]$, which uses $q_\perp$, tracks realized volatility.
Online EM does not turn the tracker into a clock.

**Read next.** {doc}`subordinator_tracking` for the derivations;
{doc}`subordinator_tracking_empirics` for the design, the meaning of
each metric and cohort, and the results that held and the ones that
did not.

**Status.** Theory 2026-08-10; S&P 500 Phases 0–3, 2026-08-13. No
package change.
