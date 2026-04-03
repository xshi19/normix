Design Decisions
================

This page explains the key design decisions in normix and the reasoning behind
them. These choices shape the API, performance characteristics, and extensibility
of the package.


Why JAX + Equinox?
------------------

normix is built on `JAX <https://jax.readthedocs.io/>`_ for computation and
`Equinox <https://docs.kidger.site/equinox/>`_ for module structure. This
combination was chosen after surveying the JAX ecosystem (TFP, Distrax, FlowJAX,
NumPyro, efax).

**Why JAX?**
The exponential family framework revolves around differentiating the log-partition
function :math:`\psi(\theta)`. In normix, a subclass only implements
:math:`\psi(\theta)` as a pure function, and ``jax.grad`` and ``jax.hessian``
automatically provide:

- Expectation parameters: :math:`\eta = \nabla\psi(\theta)`
- Fisher information: :math:`I(\theta) = \nabla^2\psi(\theta)`

This "single source of truth" design eliminates synchronisation bugs between
manually implemented gradients and the log-partition function.

**Why Equinox?**
``eqx.Module`` is a frozen dataclass that is automatically a JAX pytree.
Distributions are mathematical objects — once constructed, their parameters should
not change. Immutability matches this semantics perfectly. The EM M-step returns a
*new* model instance rather than mutating state:

.. code-block:: python

   eta = model.e_step(X)     # returns NormalMixtureEta
   new_model = model.m_step(eta)  # returns new instance

This is cleaner than mutable alternatives (Flax NNX, sklearn-style ``fit``
returning ``self``), and is the same design choice made by FlowJAX.

**Why not build on an existing distribution library?**

- **TFP**: No exponential family abstraction, very heavy dependency (~50 MB),
  TensorFlow-flavoured API
- **Distrax**: ``vmap`` support is officially "experimental and incomplete",
  hard TFP dependency, maintenance mode
- **FlowJAX**: Clean Equinox-based design, but focused on normalizing flows;
  inheriting would bring flow-oriented assumptions and dependencies
- **efax**: The only library with explicit exponential family form, but uses a
  separate class per parametrization (triples the class count), and is missing
  GIG, GH, mixtures, and EM
- **NumPyro**: Heavy PPL stack, Bayesian-focused (MCMC/SVI), not MLE/EM

No package covers normix's unique needs: GH distribution family + exponential
family structure + EM fitting. Building on Equinox directly gives full control
with minimal dependencies.


Exponential Family Architecture
-------------------------------

The core abstraction is the **log-partition triad**: three functions × two backends.

.. code-block:: text

                        JAX (JIT-able)                  CPU (numpy/scipy)
                        ──────────────                  ─────────────────
   log-partition        _log_partition_from_theta        _log_partition_cpu
   gradient             _grad_log_partition              _grad_log_partition_cpu
   Hessian              _hessian_log_partition           _hessian_log_partition_cpu

A subclass only needs to implement ``_log_partition_from_theta`` (Tier 1). The
gradient and Hessian default to ``jax.grad`` and ``jax.hessian`` (Tier 2), and
CPU versions default to wrapping the JAX implementations (Tier 3).

**Why this layered design?**

For simple distributions (Gamma, Inverse Gamma, Inverse Gaussian), ``jax.grad``
works perfectly. But the GIG distribution's log-partition involves Bessel
functions, where:

1. The JAX Hessian through ``log_kv`` is expensive (~25 Bessel calls per step
   via autodiff). An analytical 7-Bessel Hessian is much cheaper.
2. For the EM hot path, ``scipy.special.kve`` (a single C call) is ~15× faster
   than vmapping JAX's ``lax.cond`` dispatch.

The triad lets each distribution override exactly the methods that benefit from
specialisation, while inheriting sensible defaults for everything else.

**Why not efax's approach?**
efax uses a separate class per parametrization (``GammaNP``, ``GammaEP``). This
triples the number of classes and makes EM awkward — the E-step produces
expectation parameters, the M-step needs to convert them, and the fitted model
uses classical parameters. In normix, a single class handles all three
parametrizations via constructors (``from_classical``, ``from_natural``,
``from_expectation``).


Three Parametrizations
----------------------

Every exponential family distribution supports three parametrizations:

.. math::

   \text{classical} \;(\alpha, \beta, \mu, \ldots)
   \;\longleftrightarrow\;
   \text{natural}\; \theta
   \;\longleftrightarrow\;
   \text{expectation}\; \eta = \nabla\psi(\theta)

The classical parameters are human-readable (shape, rate, mean). The natural
parameters are what the exponential family density uses. The expectation
parameters are the expected sufficient statistics — and critically, the MLE is
simply :math:`\hat\eta = \frac{1}{n}\sum_i t(x_i)`.

**Why store classical parameters internally?**
Classical parameters are the most interpretable and numerically stable
representation. Natural-to-classical conversion is always analytic. The reverse
(expectation-to-natural) may require optimisation (e.g., for GIG), so storing
expectation parameters would make construction expensive.


CPU/JAX Hybrid Backend
----------------------

The EM algorithm for GH distributions has a performance bottleneck: Bessel
function evaluations in the E-step and GIG η→θ optimisation in the M-step.

Benchmarks on 468 stocks, 2552 observations show:

.. list-table::
   :header-rows: 1

   * - Phase
     - JAX (GPU)
     - CPU hybrid
     - Speedup
   * - E-step
     - ~1.1 s
     - ~0.07 s
     - ~15×
   * - M-step (GIG solve)
     - ~5–7 s
     - ~0.01 s
     - ~500×

**Why is JAX slower here?**
Each ``lax.cond`` branch in the pure-JAX Bessel triggers a separate GPU kernel
launch. For the EM hot path — which calls ``log_kv`` thousands of times per
iteration — the kernel launch overhead dominates. A single ``scipy.special.kve``
C call per element avoids this entirely.

**Why not use JAX-based Newton for GIG?**
The GIG η→θ problem is a 3-dimensional scalar optimisation. GPU kernel dispatch
overhead for such a small problem far exceeds the compute time. ``scipy``'s
L-BFGS-B, running entirely on CPU with ``scipy.special.kve``, is faster.

The ``backend`` parameter is resolved at Python level before JAX tracing begins.
When ``backend='jax'``, all code is traceable and JIT-able. When
``backend='cpu'``, the code runs eagerly — appropriate since the EM outer loop
is already a Python ``for`` loop in CPU mode.


Mixture Architecture: Joint + Marginal
---------------------------------------

The GH family is a normal variance-mean mixture:

.. math::

   X \mid Y \sim \mathcal{N}(\mu + \gamma Y, \Sigma Y), \quad Y \sim \text{subordinator}

normix separates this into two class hierarchies:

.. code-block:: text

   JointNormalMixture(ExponentialFamily)     f(x, y) — IS an exponential family
       ├── JointVarianceGamma
       ├── JointNormalInverseGamma
       ├── JointNormalInverseGaussian
       └── JointGeneralizedHyperbolic

   NormalMixture(eqx.Module)                f(x) = ∫f(x,y)dy — NOT an exp. family
       ├── VarianceGamma
       ├── NormalInverseGamma
       ├── NormalInverseGaussian
       └── GeneralizedHyperbolic

**Why two classes instead of one?**
The joint distribution :math:`f(x, y)` is an exponential family — its natural
parameters, sufficient statistics, and log-partition all have closed-form
expressions. The EM E-step works directly with the joint.

The marginal distribution :math:`f(x) = \int f(x,y)\,dy` is *not* an
exponential family. Its density involves Bessel functions and does not decompose
into the standard :math:`h(x)\exp(\theta^T t(x) - \psi(\theta))` form.

Separating them keeps each class focused: the joint knows exponential family math,
the marginal knows Bessel-based density evaluation. The marginal owns a joint
instance (accessible via ``.joint``) and delegates EM operations to it.


EM Algorithm: Model/Fitter Separation
-------------------------------------

Following the GMMX design: **the model knows math, the fitter knows iteration.**

.. code-block:: python

   from normix import GeneralizedHyperbolic
   from normix.fitting.em import BatchEMFitter

   model = GeneralizedHyperbolic.default_init(X)
   fitter = BatchEMFitter(max_iter=200, tol=1e-4)
   result = fitter.fit(model, X)

The E-step and M-step are methods on the model. The fitter controls the outer
loop, convergence checking, and timing. This separation means the same model
works with ``BatchEMFitter`` or ``IncrementalEMFitter``.

**``EMResult`` instead of a bare model:**
``fitter.fit()`` returns an ``EMResult`` containing the fitted model, diagnostics
(convergence flag, iteration count, wall-clock time), and optionally per-iteration
log-likelihoods and parameter changes.

**Convergence criterion:**
EM convergence is measured by the maximum relative change in the normal parameters
(μ, γ, L_Σ). The subordinator (GIG) parameters are excluded — they have their own
solver tolerance in the η→θ optimisation.

**Dual loop:**
When both E-step and M-step backends are ``'jax'`` and verbosity is low,
the batch EM body runs inside ``jax.lax.scan`` (fully JIT-compiled). Otherwise a
Python ``for`` loop is used, which supports CPU backends and verbose diagnostics.


Bessel Functions
----------------

Modified Bessel functions of the second kind :math:`K_\nu(z)` appear throughout
the GH family. normix provides ``log_kv(v, z)`` with two backends:

**JAX backend** (``backend='jax'``, default):
Pure-JAX, JIT-able, differentiable via ``@jax.custom_jvp``. Uses 4-regime dispatch:

1. Hankel asymptotic (large z)
2. Olver uniform expansion (large ν)
3. Small-z leading term
4. 64-point Gauss-Legendre quadrature (general case)

Derivatives: exact recurrence for ∂/∂z, central finite differences for ∂/∂ν.

**CPU backend** (``backend='cpu'``):
``scipy.special.kve``, fully vectorized NumPy. Not JIT-able, but ~15× faster for
the EM hot path. Overflow handled via asymptotic Γ-function formula.

**Why not just use TFP's Bessel?**
TFP's ``bessel_kve`` does not support gradients with respect to the order ν,
which normix needs for the GIG Hessian. normix's custom JVP provides both
∂/∂z and ∂/∂ν.
