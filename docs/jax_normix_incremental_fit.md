# Incremental Fitting for Exponential Families

**Date:** 2026-03-07

---

## 1. Why the Weighted Average Is the Right Approach

For exponential families, the MLE from $n$ observations is $\hat{\eta} = \frac{1}{n}\sum_{i=1}^n t(x_i)$. If you already have a model fitted on $n_{\text{old}}$ observations (with expectation parameters $\eta_{\text{old}}$), and new data arrives with empirical sufficient statistics $\hat{\eta}_{\text{new}} = \frac{1}{n_{\text{new}}}\sum_{j=1}^{n_{\text{new}}} t(x_j)$, then the MLE on the combined data is:

$$\eta_{\text{combined}} = \frac{n_{\text{old}} \cdot \eta_{\text{old}} + n_{\text{new}} \cdot \hat{\eta}_{\text{new}}}{n_{\text{old}} + n_{\text{new}}}$$

This is just a weighted average. No optimization needed — it's exact.

More generally, with a learning rate $\alpha \in (0, 1]$:

$$\eta_{\text{updated}} = (1 - \alpha)\,\eta_{\text{old}} + \alpha\,\hat{\eta}_{\text{new}}$$

Different choices of $\alpha$ give different behaviors:

| $\alpha$ | Interpretation |
|---|---|
| $\frac{n_{\text{new}}}{n_{\text{old}} + n_{\text{new}}}$ | Exact pooled MLE (all data equally weighted) |
| $\frac{1}{\tau_0 + t}$ | Online EM (Cappé-Moulines step size) |
| Fixed small value | Exponential forgetting (for non-stationary data) |
| $1.0$ | Ignore old model, fit fresh |

Then $\theta_{\text{updated}} = \text{eta\_to\_theta}(\eta_{\text{updated}})$.

This is **not an approximation** — for the exact pooling case, it gives the same answer as fitting on the combined dataset. The online EM update from `docs/theory/online_em.rst` is exactly this weighted average with $\alpha = \tau_t^{-1}$.

### Why not warm-start the optimizer instead?

Warm-starting (using old $\theta$ as initialization for the optimizer) is useful when `eta_to_theta` is expensive (e.g., GIG with L-BFGS-B). But it only speeds up the optimization — it doesn't change the result. The weighted average approach is better because:

1. **No optimization needed for the averaging step** — it's just array arithmetic
2. **Still needs `eta_to_theta` once** at the end, but the warm start naturally follows: the old $\theta$ is a good initial guess for the new $\eta_{\text{updated}}$ (since $\eta_{\text{updated}}$ is close to $\eta_{\text{old}}$)
3. **Mathematically exact** for the pooling case

The two ideas actually **compose**: use the weighted average for $\eta$, then use the old $\theta$ as warm-start for `eta_to_theta`.

---

## 2. Proposed API

### 2.1 `update` Method on ExponentialFamily

```python
class ExponentialFamily(eqx.Module):

    def update(self, X, *, alpha=None, n_old=None):
        """
        Incremental update: blend current model with new data.

        η_updated = (1 - α) η_old + α η_new

        Parameters
        ----------
        X : Array, shape (n_new, ...)
            New observations.
        alpha : float, optional
            Learning rate. If None, uses exact pooling:
            α = n_new / (n_old + n_new).
        n_old : int, optional
            Effective number of observations the current model represents.
            Required if alpha is None. Ignored if alpha is provided.

        Returns
        -------
        model : ExponentialFamily
            New model with updated parameters (immutable).
        """
        eta_old = self.expectation_params()
        n_new = X.shape[0]
        eta_new = jnp.mean(jax.vmap(self.__class__._sufficient_statistics_static)(X), axis=0)

        if alpha is None:
            if n_old is None:
                raise ValueError("Either alpha or n_old must be provided")
            alpha = n_new / (n_old + n_new)

        eta_updated = (1 - alpha) * eta_old + alpha * eta_new

        # Warm-start: use current theta as initial guess
        theta_old = self.natural_params()
        theta_updated = eta_to_theta(self.__class__, eta_updated, theta0=theta_old)
        return self.__class__.from_natural(theta_updated)
```

### 2.2 Usage Examples

```python
# Initial fit
model = Gamma.fit(X_train)

# Exact pooling with new data (equivalent to fitting on X_train + X_new)
model = model.update(X_new, n_old=len(X_train))

# Online learning with fixed learning rate
for X_batch in data_stream:
    model = model.update(X_batch, alpha=0.01)

# Online EM step size (Robbins-Monro)
for t, X_batch in enumerate(data_stream):
    model = model.update(X_batch, alpha=1.0 / (10.0 + t))
```

### 2.3 For Mixture Distributions (EM)

The same pattern applies to the EM M-step. The `NormalMixtureModel` version works on **expected** sufficient statistics:

```python
class NormalMixtureModel(eqx.Module):

    def update(self, X, *, alpha=None, n_old=None):
        """
        Incremental EM update: one E-step + blended M-step.

        1. E-step: compute E[t(x,Y)|x, θ_old] for each x in X
        2. Average: η_new = mean of conditional expected sufficient stats
        3. Blend: η_updated = (1-α) η_old + α η_new
        4. M-step: θ_updated = eta_to_theta(η_updated)
        """
        # E-step
        eta_new = self.expected_sufficient_statistics(X)  # mean over X

        # Blend
        eta_old = self.joint().expectation_params()
        n_new = X.shape[0]
        if alpha is None:
            alpha = n_new / (n_old + n_new)
        eta_updated = (1 - alpha) * eta_old + alpha * eta_new

        # M-step
        return self.__class__.from_expected_sufficient_statistics(eta_updated)
```

This makes the connection to online EM explicit: each `update` call is one step of the online EM from `docs/theory/online_em.rst`, with $\bar{t}(x_t|\theta_{t-1})$ replaced by the batch average of conditional expected sufficient statistics.

### 2.4 The Full Fit Spectrum

With `update`, the entire spectrum from batch to online fits into one pattern:

```python
# Batch EM (standard) — equivalent to repeated update with alpha=1
model = GH.fit(X, key=key, method='batch')

# Online EM — update one sample at a time
model = GH._initialize(X[:100], key)
for t, x_t in enumerate(X):
    model = model.update(x_t[None], alpha=1.0 / (10.0 + t))

# Mini-batch EM — update with batches
model = GH._initialize(X[:100], key)
for t in range(max_iter):
    batch = X[jax.random.choice(key, n, shape=(256,))]
    model = model.update(batch, alpha=1.0 / (10.0 + t))

# Fine-tuning — blend existing model with new data
pretrained = load_model(...)
finetuned = pretrained.update(X_new, alpha=0.3)  # 30% new, 70% old
```

---

## 3. Summary

| Approach | What it does | When to use |
|---|---|---|
| `Gamma.fit(X)` | MLE from scratch | First fit, small data |
| `model.update(X_new, n_old=n)` | Exact pooled MLE | New batch arrives, want combined MLE |
| `model.update(X_new, alpha=0.01)` | Exponential moving average | Non-stationary data, fine-tuning |
| `model.update(X_batch, alpha=1/(tau+t))` | Online EM step | Streaming, mini-batch EM |
| `GH.fit(X, method='batch')` | Full batch EM | Standard fitting |
| `GH.fit(X, method='online')` | Online EM loop | Large streaming data |

The weighted average on $\eta$ is the unifying operation. The `update` method makes it a first-class API alongside `fit`, enabling fine-tuning, streaming updates, and incremental learning — all with the same mathematical foundation.
