# Lint rules and numerical checks

Run these commands from the repository root. Ruff reads the Python source and
reports a file, line, and rule code for each finding. The second command also
checks numerical patterns that have caused bugs in normix.

```bash
uv run ruff check normix/ tests/
uv run pytest tests/test_lint_contracts.py
```

Both checks run in the default `uv run pytest tests/` suite, including in
continuous integration (CI), the tests that run on a pull request. The
project's lint scope is the package and its tests.

## Reading a Ruff finding

| Rule | What can go wrong | Example to avoid | Correction |
|---|---|---|---|
| F (names and imports) | A misspelled name can fail at runtime. An unresolved return annotation can hide a missing type dependency. Unused imports, repeated definitions, unused locals and empty formatting are also reported. | Return `"EMResult"` without importing the type anywhere in the module. Write `f"failed"` with nothing to format. | Import `EMResult` inside `if TYPE_CHECKING:` if importing it at runtime would create a cycle. Write `"failed"`. |
| E9 (source file errors) | Ruff must be able to read a file to check it. E902 reports a missing or unreadable input. | `ruff check missing.py` | Give Ruff the existing file or restore read access. |
| B (likely bugs) | `zip` normally stops at the shorter input. A test that expects any exception can pass for the wrong reason. | `zip(lower, upper)` for coordinate bounds. `pytest.raises(Exception)` for assignment to a frozen result. | `zip(lower, upper, strict=True)` checks both lengths. `pytest.raises(FrozenInstanceError)` tests immutability. |
| BLE001 (broad exception handler) | Catching `Exception` can turn a programming error into an apparently successful fit. | `except Exception: return previous` | Catch `(SolverDidNotConverge, ValueError)` at a solver boundary that can recover from those failures. |
| S110 (discarded error) | A broad handler containing only `pass` hides the cause of failure. | `except Exception: pass` | Let the error propagate, or handle the specific numerical failure. |
| S112 (silent retry) | A broad handler containing only `continue` can skip every start after a programming error. | `except Exception: continue` | In the multistart loop, retry on `SolverDidNotConverge` or `ValueError`. Other errors reach the caller. |
| NPY (NumPy APIs) | Legacy random calls use shared state. Reseeding it in a test can change the samples another test sees. | `np.random.seed(42)` followed by `np.random.gamma(3.0)` | `rng = np.random.default_rng(42)` followed by `rng.gamma(3.0)` gives the test its own repeatable draws. |
| TID251 (banned API) | A direct SciPy Bessel import in the package bypasses the shared numerical implementation. | `from scipy.special import kve` in a distribution | Import `log_kv` or `log_kv_moments` from `normix.utils.bessel`. Tests may use SciPy for comparison. |
| RUF100 (unused suppression) | A `noqa` comment tells Ruff to ignore a warning. A leftover suppression makes the reader look for an exception that no longer exists. | `return value  # noqa: F401` | `return value` |

The F checks would have flagged four unused imports removed from the joint
mixture module. The broad-handler rule would also have flagged two Generalized
Hyperbolic fallbacks that swallowed unrelated exceptions. Decision E17 limits
which failures can preserve a previous fit. The CPU multistart loop also used
to discard unexpected exceptions while trying other starts.
[Joint import cleanup](https://github.com/xshi19/normix/commit/15a90ab34ac352a936c7258281bf8a08bbe35954)
and [GH exception repair in #173](https://github.com/xshi19/normix/commit/7c5deac890c9d63db7e85137c9cde7753fbcffa7).

## Four numerical checks

### Bessel functions in production

`scipy.special.kv` returns a Bessel value and `kve` returns a scaled
value. Production callers usually need the logarithm or moments. For example,
replace `np.log(kve(v, z)) - z` with `log_kv(v, z, backend="cpu")`.
The [SciPy](https://docs.scipy.org/doc/scipy/) implementation remains an
independent reference in tests. Ruff enforces both banned names. Pytest
checks that the same import fails inside the package and passes inside a test.
The old CPU implementation imported `kve` before the shared kernel landed.
[Kernel replacement in acdad61](https://github.com/xshi19/normix/commit/acdad613ca73e39bc2d88f9e13692e328cd6a881).

### An exponential times a normal probability

A normal cumulative distribution function (CDF) gives the probability of
a value at or below a threshold. In the inverse Gaussian CDF, a large scale
times a tiny probability can be finite even when the computer evaluates the
two factors as infinity and zero. The check recognizes multiplication in
either order. Add the logarithms, then exponentiate once.

```python
# Can produce infinity times zero
jnp.exp(a) * jax.scipy.stats.norm.cdf(t)

# log_ndtr(t) computes the log probability directly
jnp.exp(a + jax.scipy.special.log_ndtr(t))
```

The inverse Gaussian implementation used the first expression before the
repair. [CDF fix in 02d2a8e](https://github.com/xshi19/normix/commit/02d2a8e307bbf6667ef1a8824706e5d2e6fb53b5).

### Precision in standalone examples

A doctest is a Python example with `>>>` prompts in a docstring. Readers
can run one outside the test suite. An example that starts with
`>>> from normix import log_kv` can inherit 32-bit arithmetic. Enable
64-bit arithmetic in the example before its first normix import.

```pycon
>>> import jax
>>> jax.config.update("jax_enable_x64", True)
>>> from normix import log_kv
```

The check reads the example statements in order. The Bessel symmetry example
compares two values to `1e-10`. In 32-bit arithmetic their difference was about
`2e-7`. [JAX precision settings](https://docs.jax.dev/en/latest/default_dtypes.html)
and [doctest fix in 8bfcaa1](https://github.com/xshi19/normix/commit/8bfcaa1fec620e424a3e50f2f4df6da0ee8cee6b).

### Derivatives on every solver attempt

A multistart solver tries several initial guesses. The caller can supply the
gradient (first derivatives) and Hessian (second derivatives) as functions.
Dropping them changes the problem the solver can evaluate. For example, CPU
Newton requires a Hessian. Every call in the retry loop must receive both
functions, including the last call after all starts fail.

```python
# Loses the caller's derivative functions
solve_bregman(f, eta, starts[0], **options)

# options contains the other solver settings
solve_bregman(f, eta, starts[0], grad_fn=grad_fn, hess_fn=hess_fn, **options)
```

The last attempt at `e51c1a7` omitted both arguments. The source check
inspects every call in the loop. A separate test forces a retry through the
public entry point and compares the actual functions by identity.
[Final retry fix in #173](https://github.com/xshi19/normix/commit/7c5deac890c9d63db7e85137c9cde7753fbcffa7).

## Which project instructions have a check?

The repository's instructions cover both executable contracts and matters
of judgment. “Partial” means that specific cases have a test.
[AGENTS.md](https://github.com/xshi19/normix/blob/master/AGENTS.md).

| AGENTS.md instruction or reading route | Enforcing check |
|---|---|
| Follow Python coding conventions | Lint in pytest: F, E9, B and NPY check selected Python and [NumPy](https://numpy.org/doc/stable/) errors. RUF100 checks suppression comments. |
| Use the shared Bessel implementation | Lint: TID251 in `normix/`. Pytest checks the production ban and the independent test-reference exception. |
| Use log-space arithmetic at extreme parameters | Pytest checks the scaled normal CDF expression. Numerical tests cover particular tail values. |
| Use float64 for distribution calculations | Pytest checks the order and value of the precision setting in standalone doctests. |
| Keep solver derivatives in natural parameter space | Pytest checks callback forwarding on every retry. Solver tests check the coordinate transformation numerically. |
| Preserve numerical failures and honor solver status (E17) | Lint: BLE001, S110 and S112. Pytest exercises recovery and propagation through actual solver calls. |
| Write specific tests and run the default suite before committing | Lint: B017 checks broad exception assertions. Pytest enforces strict markers and excludes slow, stress, integration and GPU tests by default. CI runs the default suite. |
| Remove debug prints and annotate public methods | None for complete coverage. T20 is disabled and CI does not run mypy. F821 catches unresolved annotation names. |
| Keep comments accurate and prose clear (unslop) | None for meaning, clarity or mathematical accuracy. |
| Preserve parameter conversions, derivatives, special cases, immutable updates and single-observation APIs | Partial pytest coverage through distribution, solver, marginal API and transformation tests. The {doc}`capability matrix </user_guide/capabilities>` specifies which backends support compilation. |
| Respect published documentation boundaries and build the tutorials | Docs CI runs `check_doc_links.sh` and builds Sphinx with warnings as errors. Its routine external link check is advisory. Scratch notebooks have no equivalent gate. |
| Verify mathematical claims and review numerical changes | Partial pytest coverage for numerical properties, formal coverage records and host callbacks. Mutation testing and external review are separate activities. |
| Keep plans, architecture, rules and skills current. Follow design, reading and review workflows. | None for completeness, documentation freshness or whether a review happened. The final tree also cannot establish that a regression test was run before the fix. |
| Use conventional commits and the release workflow | Release automation consumes commit history. There is no mandatory local commit-message check. |
