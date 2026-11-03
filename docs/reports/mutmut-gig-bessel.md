# Mutation testing for GIG and Bessel

mutmut 3.8.1 mutated two modules.
The modules are `normix/distributions/generalized_inverse_gaussian.py` and `normix/utils/bessel.py`.
The first file is the Generalized Inverse Gaussian (GIG) law.
The second file is the Bessel kernel `log_kv`.
The run made 3251 mutants.
The score is `(killed + timeout) / total`.

A timeout counts in the numerator.
A survivor, a mutant with no test, and a segfault stay in the denominator.
The first run scored 83.24%, which is 2706/3251.
New behavior tests then killed 110 of the 522 survivors.
The score after those tests is 86.62%, which is 2816/3251.
No library logic was changed.

| Run | Killed | Timeout | Survived | No test | Segfault | Score |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Before the new tests | 2698 | 8 | 522 | 22 | 1 | 83.24% (2706/3251) |
| After the new tests | 2808 | 8 | 412 | 22 | 1 | 86.62% (2816/3251) |

## How each original survivor was classed

The first run left 522 survivors.
A survivor is a mutant the selected tests did not kill.
Each one is a real gap, an equivalent edit, or a numerical no-show.
Twenty-three more sit on a path this run did not execute.
Those 23 are not a weak assertion in the selected tests.

| Class | Mutants |
| --- | ---: |
| Real gap, killed by a new test | 110 |
| Equivalent or no-op | 265 |
| Numerically irrelevant | 124 |
| Path not executed in this run | 23 |
| Total | 522 |

The new tests assert densities, moments, limits, and identities.
They do not assert a private control flag.
Tolerances already in the suite were not tightened to raise the score.

## Real gaps

The 110 killed mutants fall into eight behaviors.

| Behavior | Mutants |
| --- | ---: |
| Boundary draws, SciPy draws, and seeds | 43 |
| Log-partition, boundary density, and the value at 0 | 21 |
| CPU cold start from expectation parameters | 26 |
| Boundary KL divergence | 10 |
| Bessel broadcast and the moments default | 6 |
| Small-$z$ limit flags | 2 |
| Gamma-boundary mode for order in $(1, 2)$ | 1 |
| Eight-node quadrature is a probability rule | 1 |

Boundary sampling must match the Gamma or InverseGamma law.
Off the support, the draws are NaN.
A SciPy sample has the GIG mean, and the same seed repeats the draws.
The same seed check covers the PINV sampler.

On the interior, `log_prob` and `pdf` match SciPy `geninvgauss`.
At an exact Gamma or InverseGamma boundary, the log-partition matches `gammaln` on CPU and on JAX.
`log_prob(0)` is $+\infty$, $\log(a/2)$, or $-\infty$, as the order requires.

`GIG.from_expectation(..., backend='cpu')` with no warm start returns the natural parameters $\theta$.
A crash, a dropped objective, or a bad solver name fails that check.
A missing bound, a default tolerance, or a default step budget still recovers $\theta$.
Those last edits stay in the numerical class.

A Gamma or InverseGamma source has the same KL divergence as its GIG image.
For shape $\le 1$ that divergence is $+\infty$.

`log_kv` on a vector of orders matches a loop of scalar calls, and the same is true for a vector of arguments.
`log_kv_moments` with no backend matches `backend='jax'`.
Any other default string raises `ValueError`.

For $p=1/2$ and $b=0$, the point is a Gamma limit, not off the domain.
The Gamma switch must stay on for $0 < p \le 1$.

For $1 < p < 2$ and $b=0$, the mode is $2(p-1)/a$.
`quadrature(8)` returns eight positive nodes whose weights sum to 1.

## Equivalent or no-op

These 265 edits do not change a return value on the tested domain.

A dropped `dtype=float64` is still float64 in this suite.
`np.finfo(None)` is the float64 limit.
An `XX` wrapper still contains the substring the tests match.
`ValueError(None)` is still a `ValueError`.

The transformed-rejection setup stores `t` and `s`, and no reader uses them.
The rejection counter is not part of the returned sample.
`to_gig` deletes `boundary_eps`, so a new default does nothing.
JAX reads `theta[2]` when the index `theta[3]` falls past the end of a length-3 vector.

`log_kv` uses the JAX kernel unless `backend` is `'cpu'`.
The default strings `'jax'` and `'JAX'` both take that kernel.
At order 0 the two coefficient branches agree, and the two mode shifts agree.
A sign change on $\log c$ cancels after the quadrature weights are normalized.
The cross term of a mean-zero weight is unchanged if the centered deviation flips sign.

`log_base_measure` is not the value returned for $x \le 0$.
`inverse=None` still selects the CDF.
A debug print does not change the returned parameters.
A default argument that equals the value the caller passes is a no-op.

## Numerically irrelevant

These 124 edits change a procedure.
The checked number stays inside a tolerance the suite already uses.

The transformed-rejection bracket, the doubling loop, and the acceptance test can move by one step.
The draws still match the target law.
At an exact edge, the closed-form sampler and the rejection sampler have the same moments.
A warm start with a bad scale still converges.
One extra iteration, a looser residual tolerance, or an inactive bound does the same.

A CPU request that uses the JAX kernel returns the same moments and the same $\theta$.
Some small-$z$ flags swap the Gamma formula and the Bessel formula.
On the tested grid the two formulas agree.
Quadrature windows and the mode seed inside `quantile_table` stay inside the existing moment and quantile checks.

The Bessel window search, a clip of $\pm 700$, and a panel floor of $10^{-12}$ do not move the contract values.
A covariance transpose leaves a symmetric Gram inside the existing check.
The floor that builds the limit flag, from 0 to 1, does not enter $\psi$ for the tested parameters.

## Path not executed in this run

These 23 mutants are not equivalent, and a tolerance does not hide them.
The selected files never reach the assertion that rejects them.
The repository already has that assertion, so the campaign did not copy it.

Four mutants change the log-partition $\psi$ in `_eta_psi_from_pab_jax`.
Calls that return only expectation parameters $\eta$ drop $\psi$.
`test_estep_log_lik_matches_marginal` keeps $\psi$.
Replacing $\log 2$ by $\log 3$ shifts that log-likelihood by $\log 3 - \log 2$ and fails the test.

Sixteen mutants sit on the JAX cold-start arm of `from_expectation`.
Selected tests either pass a warm start or request `backend='cpu'`.
`TestSolveBregmanGIG.test_from_expectation_cold_start` calls the JAX arm with no warm start.
A crash on that arm fails that test.

Three mutants replace the Jacobian of the implicit map from $\eta$ to $\theta$.
The primal value is still the solved $\theta$.
`tests/test_expectation_autodiff.py` checks the derivative.

## Other outcomes

Eight timeouts are `asarray(None)` or a null bound array.
The time limit kills them.
They are not survivors, and they are not defects in the library.

One segfault is `_fori` when `_is_jax(None)` runs under JAX.
The score keeps that mutant in the denominator.
The failure is in the runner, not in the Bessel values.

This run marks 22 mutants as having no test.
Twenty-one edit `_psi_from_log_k`, the same helper the conjugacy test calls.
One changes the default of `boundary_eps` on `to_gig`.
That argument is deleted, so the edit is a no-op.

## Tool limits

mutmut 3 mutates the body of a function.
The module-level Gauss–Legendre nodes sit outside that scope.

`log_partition` is a compiled copy of `_log_partition_from_theta`.
The tracer records only the first test that compiles it.
`log_prob` evaluates that function at every point, so a bad formula fails the density tests.

This run skipped E-step, batch-EM, and posterior-GIG tests.
Those tests already cover $\psi$.
A copy of them in this run would not be a new check.

## Product bugs

The campaign found no product bug.
The two library modules are unchanged.

## How the run was set

Install mutmut in the local environment only.
Do not add it to the project dependencies.
Point the run at the two modules and at the behavioral tests.
Run pytest in one process.
After every `uv run`, restore `uv.lock`.

Sphinx ignores `docs/reports`, so this note is not a page on the documentation site.
