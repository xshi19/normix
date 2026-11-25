# GIG concentration symbol

## Problem

`docs/theory/gig.md` and `docs/theory/gh.md` wrote $\eta = \sqrt{ab}$ for the GIG concentration.
Elsewhere in normix, and everywhere on xshi-math, $\eta$ means $\nabla\psi(\theta)$.
The live GIG page uses $\omega = \sqrt{ab}$ and says the source called that parameter $\eta$.

## Search

The search covered `docs/`, `dev-notes/`, `normix/` docstrings and identifiers, `tests/`, and `notebooks/`.
The target was $\eta$ or `eta` used for $\sqrt{ab}$, not for an expectation parameter.

### (a) Notation, renamed here

- `docs/theory/gig.md`. The scale-concentration sentence now uses $\omega$. The old $(p, \delta, \eta)$ density had prefactor $\delta^p$. Substitution of $\delta = \sqrt{b/a}$ gives $\delta^{-p}$. That display is removed. The corrected density is on the xshi-math GIG page.
- `docs/theory/gh.md`. The alternative parameterization display is removed. The scale orbit is written with invariant $\omega$.
- `dev-notes/research/formal_verification/01_lean_math_feasibility.md`. The finding described the live page as using $\eta = \sqrt{ab}$. It now records that the library page no longer does.

`docs/theory/varentropy.md` already used $z = \sqrt{ab}$ for the Bessel argument. That local name stays, so the tutorial display and the tests still match. One sentence there maps $z$ to $\omega$.

### (b) Code and API identifiers

None. No Python identifier, public argument, test, or notebook uses `eta` for $\sqrt{ab}$.

Every `eta` argument found is an expectation parameter $\nabla\psi$.
These public methods are not a rename list:

- `GeneralizedInverseGaussian.from_expectation(eta)`
- `MarginalMixture.m_step(eta)`
- `JointNormalMixture.from_expectation`

Checked near-misses. None of them is a public name for $\sqrt{ab}$.

| Location | What it is | Public API? |
|---|---|---|
| `normix/distributions/generalized_inverse_gaussian.py:34-43` | $\eta$-rescaling of the expectation vector. $\sqrt{\eta_2\eta_3}$ is a component of $\tilde\eta$ after that rescaling. | No. `from_expectation` still takes $\nabla\psi$. |
| `tests/test_jax_distributions.py:223` | SciPy `geninvgauss` uses `b = \sqrt{ab}` and `scale = \sqrt{b/a}`. | No. Test oracle. The arguments are not named `eta`. |
| `tests/test_distributions_vs_scipy.py:243` | Same SciPy arguments. | No. |
| `sqrt_ab`, `z`, and `t` in code, tests, and `docs/tutorials/stats/03_varentropy.md` | The Bessel argument $\sqrt{ab}$. | Local names, not `eta`. |

## Decision

Use $\omega$ for $\sqrt{ab}$. Keep $\delta = \sqrt{b/a}$ for the scale. Keep $\eta$ for $\nabla\psi$.

The live xshi-math GIG page already uses $\omega$.
`gradient_fitting_comparison.md` already uses the same $\omega$ for this quantity.
Keeping $\eta$ would leave two meanings in the theory pages.
Renaming the varentropy argument $z$ would split it from the tutorial and the tests.

No code identifier was renamed. Reopen this choice if xshi-math stops using $\omega$ for $\sqrt{ab}$.
Authority for the split is the instruction that derivations live on xshi-math and that this symbol should match that page.

## References

- <https://xshi19.github.io/math/normix-theory/normix-generalized-inverse-gaussian/>
- `gradient_fitting_comparison.md`, scale and concentration paragraph
- `dev-notes/design/design.md`, row F11
