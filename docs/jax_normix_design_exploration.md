# JAX-Based Distribution Design: Equinox & FlowJAX Deep Dive

## Table of Contents

1. [Part 1: Equinox (`eqx.Module`)](#part-1-equinox-eqxmodule)
2. [Part 2: Equinox vs Flax NNX](#part-2-equinox-vs-flax-nnx)
3. [Part 3: FlowJAX `AbstractDistribution`](#part-3-flowjax-abstractdistribution)
4. [Part 4: FlowJAX Conditional Distributions](#part-4-flowjax-conditional-distributions)
5. [Design Implications for normix-JAX](#design-implications-for-normix-jax)

---

## Part 1: Equinox (`eqx.Module`)

### 1.1 What is Equinox?

Equinox (by Patrick Kidger) is a JAX library that bridges the gap between JAX's functional paradigm and PyTorch-like object-oriented model definition. The core idea: **models are PyTrees that happen to have methods**.

`eqx.Module` is the base class. It is simultaneously:
- A **Python dataclass** (auto-generates `__init__`, field declarations)
- A **JAX pytree** (auto-registered, works with `jax.jit`, `jax.grad`, `jax.vmap`)
- An **ABC** (supports `abc.abstractmethod`, `AbstractVar`, `AbstractClassVar`)

### 1.2 How `eqx.Module` Works Internally

From the actual source (`equinox/_module/_module.py`):

```python
@dataclass_transform(field_specifiers=(dataclasses.field, field))
class _ModuleMeta(BetterABCMeta):
    def __new__(mcs, name, bases, namespace, *, is_abstract=False, strict=False, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Create a dataclass __init__ if user didn't provide one
        has_dataclass_init = "__init__" not in cls.__dict__
        cls = better_dataclass(eq=False, repr=False, init=has_dataclass_init)(cls)

        fields = dataclasses.fields(cls)

        # Generate optimized flatten/unflatten functions
        flatten_func, flatten_with_keys_func, unflatten_func = (
            generate_flatten_functions(cls, fields)
        )

        # Register as JAX pytree
        jtu.register_pytree_with_keys(
            cls,
            flatten_with_keys=flatten_with_keys_func,
            flatten_func=flatten_func,
            unflatten_func=ft.partial(unflatten_func, cls),
        )
        return cls
```

**Key mechanism**: `_ModuleMeta` is a metaclass. Every time you write `class Foo(eqx.Module)`, the metaclass:
1. Converts the class into a dataclass (`better_dataclass`)
2. Generates optimized `tree_flatten` / `tree_unflatten` functions
3. Calls `jtu.register_pytree_with_keys()` to register the class as a JAX pytree

### 1.3 Parameter Storage

Parameters are stored as **named fields**, exactly like a Python dataclass:

```python
class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_features, out_features, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_features, in_features))
        self.bias = jax.random.normal(bkey, (out_features,))
```

**Difference from a plain dataclass**: A plain `@dataclass` is not a JAX pytree. `eqx.Module` automatically registers it as one. The fields become pytree children (leaves), unless marked `static=True`.

### 1.4 Immutability

`eqx.Module` is **immutable after `__init__`**. From the source:

```python
class Module(Hashable, metaclass=_ModuleMeta):
    def __setattr__(self, name, value):
        if self in _currently_initialising and name in _module_info[type(self)].names_set:
            object.__setattr__(self, name, value)  # allowed during __init__
            return
        raise dataclasses.FrozenInstanceError(f"cannot assign to field '{name}'")
```

- During `__init__`, field assignment is allowed (tracked via `_currently_initialising` set)
- After `__init__` completes, any `self.x = ...` raises `FrozenInstanceError`
- This matches JAX's functional paradigm: you don't mutate; you create new copies

### 1.5 PyTree Registration: Leaves vs Static

When a Module is flattened as a pytree, the generated functions separate fields into:

- **Dynamic (leaves)**: Regular fields — these are the pytree children. JAX sees them, traces them, differentiates through them.
- **Static (auxiliary data)**: Fields marked with `eqx.field(static=True)` — these become part of the pytree *treedef* (structure). JAX does not trace them; they must be hashable.

From `equinox/_module/_flatten.py`:

```python
def generate_flatten_functions(cls, fields):
    _dynamic_fs, _static_fs = [], []
    for f in fields:
        if f.metadata.get("static", False):
            _static_fs.append(f.name)
        else:
            _dynamic_fs.append(f.name)
    # ... generates exec'd flatten/unflatten functions
```

**Concrete example**:
```python
class MyDist(eqx.Module):
    mu: jax.Array                              # leaf (dynamic)
    sigma: jax.Array                           # leaf (dynamic)
    dim: int = eqx.field(static=True)          # static (treedef)
    name: str = eqx.field(static=True)         # static (treedef)

dist = MyDist(mu=jnp.array([1.0]), sigma=jnp.array([2.0]), dim=1, name="normal")
leaves, treedef = jax.tree_util.tree_flatten(dist)
# leaves = [Array([1.]), Array([2.])]
# treedef encodes: dim=1, name="normal", field structure
```

### 1.6 `eqx.field(static=True)`

From the source (`equinox/_module/_field.py`):

```python
def field(*, converter=None, static=False, **kwargs):
    metadata = {}
    if converter is not None:
        metadata["converter"] = converter
    if static:
        metadata["static"] = True
    return dataclasses.field(metadata=metadata, **kwargs)
```

**When to use `static=True`**:
- Python types that aren't JAX arrays: `int`, `str`, `bool`, `tuple[int, ...]`
- Activation functions (`Callable`)
- Configuration values that never change
- The `shape` and `cond_shape` attributes in FlowJAX

**When NOT to use it**:
- JAX arrays (a warning is raised if you mark an array as static)
- Anything you want to differentiate through

### 1.7 Functional Updates with `eqx.tree_at`

Since modules are immutable, you create new ones with changed fields:

```python
import equinox as eqx
import jax.numpy as jnp

class Normal(eqx.Module):
    loc: jax.Array
    scale: jax.Array

dist = Normal(loc=jnp.array(0.0), scale=jnp.array(1.0))

# Create a new distribution with different loc
new_dist = eqx.tree_at(lambda d: d.loc, dist, jnp.array(5.0))
# new_dist.loc = 5.0, new_dist.scale = 1.0 (unchanged)

# Using replace_fn (applies function to current value)
doubled_scale = eqx.tree_at(lambda d: d.scale, dist, replace_fn=lambda s: 2 * s)

# Replace multiple fields at once
new_dist = eqx.tree_at(
    lambda d: (d.loc, d.scale),
    dist,
    (jnp.array(3.0), jnp.array(0.5)),
)
```

### 1.8 Filter Transforms

#### `eqx.filter_jit`

Standard `jax.jit` requires you to manually specify `static_argnums` for non-array arguments. `eqx.filter_jit` does this automatically:

```python
# With jax.jit: you must manually mark non-array args
@jax.jit
def f(model, x):  # model contains both arrays and non-arrays!
    return model(x)  # FAILS: non-arrays aren't valid JIT inputs

# With eqx.filter_jit: automatic
@eqx.filter_jit
def f(model, x):
    return model(x)  # WORKS: arrays traced, non-arrays static
```

**How it works** (from source): `filter_jit` partitions every argument into `is_array` (dynamic, traced) and non-array (static, hashed). It then calls `jax.jit` with the array parts as dynamic inputs and everything else as static. On output, it recombines.

#### `eqx.filter_grad`

Standard `jax.grad` differentiates all arguments. `eqx.filter_grad` automatically differentiates only floating-point arrays:

```python
@eqx.filter_grad
def loss(model, x, y):  # model is a Module with arrays + non-arrays
    return jnp.mean((model(x) - y) ** 2)

grads = loss(model, x, y)
# grads is a Module with same structure as model,
# but non-float leaves are None, float leaves have gradients
```

From the source (`equinox/_ad.py`):

```python
class _ValueAndGradWrapper(Module):
    def __call__(self, *args, **kwargs):
        @ft.partial(jax.value_and_grad, has_aux=self._has_aux)
        def fun_value_and_grad(_diff_x, _nondiff_x, *_args, **_kwargs):
            _x = combine(_diff_x, _nondiff_x)
            return self._fun(_x, *_args, **_kwargs)

        x, *args = args
        diff_x, nondiff_x = partition(x, is_inexact_array)
        return fun_value_and_grad(diff_x, nondiff_x, *args, **kwargs)
```

#### `eqx.filter_vmap`

Vmaps all array leaves, broadcasts everything else:

```python
@eqx.filter_vmap
def batched_log_prob(model, x):
    return model.log_prob(x)  # x has leading batch dim

# Or with explicit axis control
@eqx.filter_vmap(in_axes=(None, 0))  # broadcast model, vmap x
def batched_log_prob(model, x):
    return model.log_prob(x)
```

### 1.9 `eqx.partition` and `eqx.combine`

These split and recombine pytrees based on a filter:

```python
# Separate trainable from non-trainable parameters
trainable, frozen = eqx.partition(model, eqx.is_inexact_array)
# trainable: same structure as model, but non-array leaves are None
# frozen: same structure, but array leaves are None

# Reconstruct
full_model = eqx.combine(trainable, frozen)

# Use case: freeze specific layers
filter_spec = jax.tree_util.tree_map(lambda _: True, model)
filter_spec = eqx.tree_at(
    lambda m: m.layers[0].weight,
    filter_spec,
    replace=False,  # freeze this weight
)
trainable, frozen = eqx.partition(model, filter_spec)
```

### 1.10 Concrete Example: Distribution as eqx.Module

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

class Gamma(eqx.Module):
    """Gamma distribution parameterized by shape (alpha) and rate (beta)."""
    _alpha: Array
    _beta: Array

    def __init__(self, alpha: float, beta: float):
        self._alpha = jnp.asarray(alpha, dtype=float)
        self._beta = jnp.asarray(beta, dtype=float)

    def log_prob(self, x: Array) -> Array:
        alpha, beta = self._alpha, self._beta
        return (
            alpha * jnp.log(beta)
            - jax.scipy.special.gammaln(alpha)
            + (alpha - 1) * jnp.log(x)
            - beta * x
        )

    def sample(self, key: PRNGKeyArray, shape: tuple = ()) -> Array:
        return jr.gamma(key, self._alpha, shape=shape) / self._beta

    @property
    def mean(self) -> Array:
        return self._alpha / self._beta

    @property
    def variance(self) -> Array:
        return self._alpha / (self._beta ** 2)

# Usage:
dist = Gamma(2.0, 1.0)

# Works with JAX transforms:
@eqx.filter_jit
def compute(dist, x):
    return dist.log_prob(x)

@eqx.filter_grad
def grad_log_prob(dist, x):
    return dist.log_prob(x).sum()

x = jnp.array([1.0, 2.0, 3.0])
lp = compute(dist, x)
grads = grad_log_prob(dist, x)
# grads is a Gamma with gradients in _alpha and _beta

# Functional update:
new_dist = eqx.tree_at(lambda d: d._alpha, dist, jnp.array(5.0))
```

---

## Part 2: Equinox vs Flax NNX

### 2.1 What is `flax.nnx.Module`?

Flax NNX is Google's newer neural network library for JAX. Its `Module` is designed to feel like PyTorch:

```python
import flax.nnx as nnx

class Linear(nnx.Module):
    def __init__(self, in_features, out_features, *, rngs):
        self.weight = nnx.Param(jax.random.normal(rngs.params(), (out_features, in_features)))
        self.bias = nnx.Param(jnp.zeros(out_features))

    def __call__(self, x):
        return x @ self.weight.value.T + self.bias.value
```

### 2.2 Key Differences

| Feature | `eqx.Module` | `flax.nnx.Module` |
|---|---|---|
| **Mutability** | Immutable after `__init__` | Mutable (reference semantics) |
| **State** | State = the pytree itself | State stored in `Variable` wrappers (`Param`, `BatchStat`, etc.) |
| **Updates** | Functional: `eqx.tree_at(...)` returns new module | In-place: `self.weight.value = new_val` |
| **Pytree** | Module IS the pytree | Module is a Python object graph; `nnx.split()`/`nnx.merge()` convert to/from pytrees |
| **JIT** | `eqx.filter_jit(fn)(module, x)` | `@nnx.jit` or manual `split`/`merge` |
| **Grad** | `eqx.filter_grad` auto-filters | `nnx.grad` with `wrt` parameter |
| **Paradigm** | Functional (JAX-native) | Object-oriented (PyTorch-like) |
| **Overhead** | Minimal (just a dataclass + pytree) | Extra indirection through Variable wrappers |
| **Performance** | Generally faster | Reported ~3x slower in some benchmarks |

### 2.3 Mutability Comparison

**Equinox (immutable)**:
```python
# Cannot do this:
dist.mu = new_mu  # raises FrozenInstanceError

# Must do:
new_dist = eqx.tree_at(lambda d: d.mu, dist, new_mu)
```

**Flax NNX (mutable)**:
```python
# Can directly mutate:
model.weight.value = new_weight  # works fine
model.count += 1  # works fine
```

### 2.4 Which is Better for a Distribution Library?

**Equinox is the clear choice** for a probability distribution library. Reasons:

1. **Immutability matches math**: A distribution with parameters $(\mu, \sigma)$ is a mathematical object. Changing $\mu$ gives you a *different* distribution, not a mutated one. Equinox's functional update model (`tree_at`) is the right semantic.

2. **Minimal overhead**: `eqx.Module` is literally just a frozen dataclass registered as a pytree. No `Variable` wrappers, no `Param` types, no state management boilerplate.

3. **Direct JAX interop**: An `eqx.Module` IS a pytree. You can pass it to `jax.jit`, `jax.grad`, `jax.vmap` directly. NNX requires split/merge gymnastics.

4. **FlowJAX precedent**: The most successful JAX distribution library (FlowJAX) uses Equinox, not Flax.

5. **Parameter constraints via paramax**: The `Parameterize` wrapper (which IS an `eqx.Module`) provides elegant constrained parameter handling without mutability.

6. **NNX solves problems we don't have**: NNX's mutability is designed for stateful training loops (batch norm running stats, RNG state). Distributions are stateless.

### 2.5 Trade-offs

| Equinox Advantage | NNX Advantage |
|---|---|
| Simpler mental model for math objects | Familiar to PyTorch users |
| Better performance | Easier state management for training |
| Direct JAX pytree | Google-backed, part of official Flax |
| FlowJAX ecosystem | NNX optimizers integration |
| No wrapper boilerplate | Mutable makes debugging easier |

---

## Part 3: FlowJAX `AbstractDistribution` Deep Dive

### 3.1 Full `AbstractDistribution` Source Code

From `flowjax/distributions.py` (FlowJAX v19.1.0):

```python
class AbstractDistribution(eqx.Module):
    """Abstract distribution class.

    Distributions are registered as JAX PyTrees (as they are equinox modules), and as
    such they are compatible with normal JAX operations.

    Concrete subclasses can be implemented as follows:

    - Inherit from :class:`AbstractDistribution`.
    - Define the abstract attributes ``shape`` and ``cond_shape``.
      ``cond_shape`` should be ``None`` for unconditional distributions.
    - Define the abstract method ``_sample`` which returns a single sample
      with shape ``dist.shape``, (given a single conditioning variable, if needed).
    - Define the abstract method ``_log_prob``, returning a scalar log probability
      of a single sample, (given a single conditioning variable, if needed).

    The abstract class then defines vectorized versions with shape checking for the
    public API. See the source code for :class:`StandardNormal` for a simple concrete
    example.

    Attributes:
        shape: Tuple denoting the shape of a single sample from the distribution.
        cond_shape: Tuple denoting the shape of an instance of the conditioning
            variable. This should be None for unconditional distributions.
    """

    shape: AbstractVar[tuple[int, ...]]
    cond_shape: AbstractVar[tuple[int, ...] | None]

    @abstractmethod
    def _log_prob(self, x: Array, condition: Array | None = None) -> Array:
        """Evaluate the log probability of point x.

        This method should be be valid for inputs with shapes matching
        ``distribution.shape`` and ``distribution.cond_shape`` for conditional
        distributions (i.e. it defines the method for unbatched inputs).
        """

    @abstractmethod
    def _sample(self, key: PRNGKeyArray, condition: Array | None = None) -> Array:
        """Sample a point from the distribution.

        This method should return a single sample with shape matching
        ``distribution.shape``.
        """

    def _sample_and_log_prob(self, key: PRNGKeyArray, condition: Array | None = None):
        """Sample a point from the distribution, and return its log probability."""
        x = self._sample(key, condition)
        return x, self._log_prob(x, condition)

    def log_prob(self, x: ArrayLike, condition: ArrayLike | None = None) -> Array:
        """Evaluate the log probability.

        Uses numpy-like broadcasting if additional leading dimensions are passed.
        """
        self = unwrap(self)
        x = arraylike_to_array(x, err_name="x", dtype=float)
        if self.cond_shape is not None:
            condition = arraylike_to_array(condition, err_name="condition", dtype=float)
        return self._vectorize(self._log_prob)(x, condition)

    def sample(
        self,
        key: PRNGKeyArray,
        sample_shape: tuple[int, ...] = (),
        condition: ArrayLike | None = None,
    ) -> Array:
        """Sample from the distribution."""
        self = unwrap(self)
        if self.cond_shape is not None:
            condition = arraylike_to_array(condition, err_name="condition")
        keys = self._get_sample_keys(key, sample_shape, condition)
        return self._vectorize(self._sample)(keys, condition)

    def sample_and_log_prob(
        self,
        key: PRNGKeyArray,
        sample_shape: tuple[int, ...] = (),
        condition: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        """Sample the distribution and return the samples with their log probs."""
        self = unwrap(self)
        if self.cond_shape is not None:
            condition = arraylike_to_array(condition, err_name="condition")
        keys = self._get_sample_keys(key, sample_shape, condition)
        return self._vectorize(self._sample_and_log_prob)(keys, condition)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def cond_ndim(self) -> None | int:
        return None if self.cond_shape is None else len(self.cond_shape)

    def _vectorize(self, method: Callable) -> Callable:
        """Returns a vectorized version of the distribution method."""
        maybe_cond = [] if self.cond_shape is None else [self.cond_shape]
        in_shapes = {
            "_sample_and_log_prob": [()] + maybe_cond,
            "_sample": [()] + maybe_cond,
            "_log_prob": [self.shape] + maybe_cond,
        }
        out_shapes = {
            "_sample_and_log_prob": [self.shape, ()],
            "_sample": [self.shape],
            "_log_prob": [()],
        }
        in_shapes, out_shapes = in_shapes[method.__name__], out_shapes[method.__name__]

        def _check_shapes(method):
            @wraps(method)
            def _wrapper(*args, **kwargs):
                bound = inspect.signature(method).bind(*args, **kwargs)
                for in_shape, (name, arg) in zip(in_shapes, bound.arguments.items(), strict=False):
                    if arg.shape != in_shape:
                        raise ValueError(
                            f"Expected trailing dimensions matching {in_shape} for "
                            f"{name}; got {arg.shape}.",
                        )
                return method(*args, **kwargs)
            return _wrapper

        signature = _get_ufunc_signature(in_shapes, out_shapes)
        ex = frozenset([1]) if self.cond_shape is None else frozenset()
        return jnp.vectorize(_check_shapes(method), signature=signature, excluded=ex)

    def _get_sample_keys(self, key, sample_shape, condition):
        if not dtypes.issubdtype(key.dtype, dtypes.prng_key):
            raise TypeError("New-style typed JAX PRNG keys required.")
        if self.cond_ndim is not None:
            leading_cond_shape = condition.shape[: -self.cond_ndim or None]
        else:
            leading_cond_shape = ()
        key_shape = sample_shape + leading_cond_shape
        key_size = prod(key_shape)
        return jr.split(key, key_size).reshape(key_shape)
```

### 3.2 Vectorization Mechanism: `_log_prob` → `log_prob`

The key is `_vectorize()`, which uses **`jnp.vectorize`** (NumPy-style generalized ufunc signatures):

1. `_log_prob(self, x, condition)` operates on **unbatched** input: `x` has shape `self.shape`, returns scalar
2. `_vectorize` constructs a **ufunc signature** like `"(3),(2)->()"` meaning: input1 has core shape `(3,)`, input2 has core shape `(2,)`, output is scalar
3. `jnp.vectorize` then broadcasts over **leading dimensions** automatically

**Example for a 3D distribution with 2D conditioning**:
```
_log_prob signature: "(3),(2)->()"
  - x must end with shape (3,)
  - condition must end with shape (2,)
  - output is scalar per (x, condition) pair

Call: dist.log_prob(x_batch, cond_batch)
  - x_batch shape: (100, 3)  → 100 leading dims
  - cond_batch shape: (100, 2) → broadcasts
  - output shape: (100,)
```

For unconditional distributions, `condition` is excluded via `excluded=frozenset([1])` (the second arg index).

### 3.3 Sampling Vectorization: `_sample` → `sample`

Sampling is vectorized similarly but requires key management:

1. `_get_sample_keys` pre-splits the PRNG key into the right shape: `sample_shape + leading_cond_shape`
2. `_vectorize(self._sample)` maps over the key array with signature `"()->(3)"` (scalar key → sample of shape `(3,)`)
3. Each call to `_sample` gets one independent key

### 3.4 Class Hierarchy: `Normal` → `AbstractLocScaleDistribution` → `AbstractTransformed` → `AbstractDistribution`

#### `AbstractDistribution` (base)
```
- shape: AbstractVar[tuple[int, ...]]
- cond_shape: AbstractVar[tuple[int, ...] | None]
- _log_prob(x, condition) → scalar        [ABSTRACT]
- _sample(key, condition) → sample         [ABSTRACT]
- log_prob(x, condition) → batched         [vectorized wrapper]
- sample(key, shape, condition) → batched  [vectorized wrapper]
```

#### `AbstractTransformed(AbstractDistribution)` (transformed distributions)
```
- base_dist: AbstractVar[AbstractDistribution]
- bijection: AbstractVar[AbstractBijection]

- shape → base_dist.shape (property)
- cond_shape → merge(bijection.cond_shape, base_dist.cond_shape) (property)

- _log_prob(x, condition):
    z, log_abs_det = bijection.inverse_and_log_det(x, condition)
    return base_dist._log_prob(z, condition) + log_abs_det

- _sample(key, condition):
    z = base_dist._sample(key, condition)
    return bijection.transform(z, condition)

- _sample_and_log_prob(key, condition):
    z, log_prob_z = base_dist._sample_and_log_prob(key, condition)
    x, log_det = bijection.transform_and_log_det(z, condition)
    return x, log_prob_z - log_det
```

#### `AbstractLocScaleDistribution(AbstractTransformed)` (loc-scale family)
```
- base_dist: AbstractVar[AbstractDistribution]
- bijection: AbstractVar[Affine]

- loc → bijection.loc (property)
- scale → unwrap(bijection.scale) (property)
```

#### `Normal(AbstractLocScaleDistribution)` (concrete)
```python
class Normal(AbstractLocScaleDistribution):
    base_dist: StandardNormal
    bijection: Affine

    def __init__(self, loc=0, scale=1):
        self.base_dist = StandardNormal(
            jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale)),
        )
        self.bijection = Affine(loc=loc, scale=scale)
```

**The chain**: `Normal` stores a `StandardNormal` (which knows `_log_prob` and `_sample` for $\mathcal{N}(0,1)$) and an `Affine` bijection ($y = \text{scale} \cdot x + \text{loc}$). The `AbstractTransformed` machinery composes them:
- `log_prob(x)` → invert affine → get $z$ → `StandardNormal._log_prob(z)` + log|det|
- `sample(key)` → `StandardNormal._sample(key)` → apply affine

### 3.5 `AbstractTransformed`: Composing Base + Bijection

Full source:

```python
class AbstractTransformed(AbstractDistribution):
    base_dist: AbstractVar[AbstractDistribution]
    bijection: AbstractVar[AbstractBijection]

    def __check_init__(self):
        if (self.base_dist.cond_shape is not None
            and self.bijection.cond_shape is not None
            and self.base_dist.cond_shape != self.bijection.cond_shape):
            raise ValueError("Mismatched cond_shapes.")
        if self.base_dist.shape != self.bijection.shape:
            raise ValueError("Mismatched shapes.")

    def _log_prob(self, x, condition=None):
        z, log_abs_det = self.bijection.inverse_and_log_det(x, condition)
        p_z = self.base_dist._log_prob(z, condition)
        log_prob = p_z + log_abs_det
        return jnp.where(jnp.isnan(log_prob), -jnp.inf, log_prob)

    def _sample(self, key, condition=None):
        base_sample = self.base_dist._sample(key, condition)
        return self.bijection.transform(base_sample, condition)

    def _sample_and_log_prob(self, key, condition=None):
        base_sample, log_prob_base = self.base_dist._sample_and_log_prob(key, condition)
        sample, forward_log_dets = self.bijection.transform_and_log_det(base_sample, condition)
        return sample, log_prob_base - forward_log_dets

    def merge_transforms(self):
        """Unnest nested transformed distributions."""
        if not isinstance(self.base_dist, AbstractTransformed):
            return self
        base_dist = self.base_dist
        bijections = [self.bijection]
        while isinstance(base_dist, AbstractTransformed):
            bijections.append(base_dist.bijection)
            base_dist = base_dist.base_dist
        bijection = Chain(list(reversed(bijections))).merge_chains()
        return Transformed(base_dist, bijection)

    @property
    def shape(self):
        return self.base_dist.shape

    @property
    def cond_shape(self):
        return merge_cond_shapes((self.bijection.cond_shape, self.base_dist.cond_shape))
```

### 3.6 How `unwrap` Works (paramax)

Full source of `paramax.wrappers`:

```python
class AbstractUnwrappable(eqx.Module, Generic[T]):
    """An abstract class representing an unwrappable object."""
    @abstractmethod
    def unwrap(self) -> T:
        """Returns the unwrapped pytree, assuming no wrapped subnodes exist."""
        pass


def unwrap(tree: PyTree):
    """Map across a PyTree and unwrap all AbstractUnwrappable nodes."""
    def _unwrap(tree, *, include_self: bool):
        def _map_fn(leaf):
            if isinstance(leaf, AbstractUnwrappable):
                return _unwrap(leaf, include_self=False).unwrap()
            return leaf

        def is_leaf(x):
            is_unwrappable = isinstance(x, AbstractUnwrappable)
            included = include_self or x is not tree
            return is_unwrappable and included

        return jax.tree_util.tree_map(f=_map_fn, tree=tree, is_leaf=is_leaf)

    return _unwrap(tree, include_self=True)


class Parameterize(AbstractUnwrappable[T]):
    """Unwrap an object by calling fn with args and kwargs."""
    fn: Callable[..., T]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = tuple(args)
        self.kwargs = kwargs

    def unwrap(self) -> T:
        return self.fn(*self.args, **self.kwargs)
```

**Usage pattern for constrained parameters**:

```python
from paramax import Parameterize
from jax.nn import softplus

class Gamma(eqx.Module):
    # Store unconstrained parameter, apply softplus on unwrap
    concentration: Array | Parameterize[Array]

    def __init__(self, concentration):
        # Store inv_softplus(concentration) internally
        self.concentration = Parameterize(softplus, inv_softplus(concentration))
        # On unwrap: softplus(inv_softplus(conc)) = conc (positive!)

    def _log_prob(self, x, condition=None):
        # self is already unwrapped when this is called (done in log_prob())
        return jax.scipy.stats.gamma.logpdf(x, self.concentration).sum()
```

The `unwrap` call in `log_prob()` (`self = unwrap(self)`) recursively transforms the entire distribution tree, applying `softplus` to any `Parameterize` nodes. This ensures all parameters satisfy their constraints during computation while keeping unconstrained values for gradient-based optimization.

### 3.7 Concrete Distribution Source Code

#### `StandardNormal`
```python
class StandardNormal(AbstractDistribution):
    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def _log_prob(self, x, condition=None):
        return jstats.norm.logpdf(x).sum()

    def _sample(self, key, condition=None):
        return jr.normal(key, self.shape)
```

#### `Normal` (via affine transform)
```python
class Normal(AbstractLocScaleDistribution):
    base_dist: StandardNormal
    bijection: Affine

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        self.base_dist = StandardNormal(
            jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale)),
        )
        self.bijection = Affine(loc=loc, scale=scale)
```

#### `Gamma` (via scale transform)
```python
class _StandardGamma(AbstractDistribution):
    concentration: Array | AbstractUnwrappable[Array]
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    def __init__(self, concentration):
        self.concentration = Parameterize(softplus, inv_softplus(concentration))
        self.shape = jnp.shape(concentration)

    def _sample(self, key, condition=None):
        return jr.gamma(key, self.concentration)

    def _log_prob(self, x, condition=None):
        return jstats.gamma.logpdf(x, self.concentration).sum()


class Gamma(AbstractTransformed):
    base_dist: _StandardGamma
    bijection: Scale

    def __init__(self, concentration, scale):
        concentration, scale = jnp.broadcast_arrays(concentration, scale)
        self.base_dist = _StandardGamma(concentration)
        self.bijection = Scale(scale)
```

#### `MultivariateNormal` (via triangular affine)
```python
class MultivariateNormal(AbstractTransformed):
    base_dist: StandardNormal
    bijection: TriangularAffine

    def __init__(self, loc, covariance):
        self.bijection = TriangularAffine(loc, linalg.cholesky(covariance))
        self.base_dist = StandardNormal(self.bijection.shape)

    @property
    def loc(self):
        return self.bijection.loc

    @property
    def covariance(self):
        cholesky = unwrap(self.bijection.triangular)
        return cholesky @ cholesky.T
```

#### `Affine` bijection (used by Normal)
```python
class Affine(AbstractBijection):
    """Elementwise affine: y = scale * x + loc"""
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    loc: Array
    scale: Array | AbstractUnwrappable[Array]

    def __init__(self, loc=0, scale=1):
        self.loc, scale = jnp.broadcast_arrays(
            *(arraylike_to_array(a, dtype=float) for a in (loc, scale)),
        )
        self.shape = scale.shape
        self.scale = Parameterize(softplus, inv_softplus(scale))

    def transform_and_log_det(self, x, condition=None):
        return x * self.scale + self.loc, jnp.log(jnp.abs(self.scale)).sum()

    def inverse_and_log_det(self, y, condition=None):
        return (y - self.loc) / self.scale, -jnp.log(jnp.abs(self.scale)).sum()
```

#### `TriangularAffine` bijection (used by MultivariateNormal)
```python
class TriangularAffine(AbstractBijection):
    """Ax + b where A is lower/upper triangular."""
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    loc: Array
    triangular: Array | AbstractUnwrappable[Array]
    lower: bool

    def __init__(self, loc, arr, *, lower=True):
        loc, arr = (arraylike_to_array(a, dtype=float) for a in (loc, arr))
        dim = arr.shape[0]
        arr = jnp.fill_diagonal(arr, inv_softplus(jnp.diag(arr)), inplace=False)

        @partial(jnp.vectorize, signature="(d,d)->(d,d)")
        def _to_triangular(arr):
            tri = jnp.tril(arr) if lower else jnp.triu(arr)
            return jnp.fill_diagonal(tri, softplus(jnp.diag(tri)), inplace=False)

        self.triangular = Parameterize(_to_triangular, arr)
        self.lower = lower
        self.shape = (dim,)
        self.loc = jnp.broadcast_to(loc, (dim,))

    def transform_and_log_det(self, x, condition=None):
        y = self.triangular @ x + self.loc
        return y, jnp.log(jnp.abs(jnp.diag(self.triangular))).sum()

    def inverse_and_log_det(self, y, condition=None):
        x = solve_triangular(self.triangular, y - self.loc, lower=self.lower)
        return x, -jnp.log(jnp.abs(jnp.diag(self.triangular))).sum()
```

---

## Part 4: FlowJAX Conditional Distributions

### 4.1 How `cond_shape` Works

Every `AbstractDistribution` has a `cond_shape` attribute:
- `None` → unconditional distribution
- `tuple[int, ...]` → conditional distribution; this is the shape of a single conditioning variable

```python
class StandardNormal(AbstractDistribution):
    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None       # unconditional

class ConditionalFlow(AbstractTransformed):
    # cond_shape is a property derived from base_dist and bijection
    @property
    def cond_shape(self):
        return merge_cond_shapes((self.bijection.cond_shape, self.base_dist.cond_shape))
```

### 4.2 How `_log_prob(x, condition)` Works with Conditions

The condition flows through the entire chain:

```python
# In AbstractTransformed._log_prob:
def _log_prob(self, x, condition=None):
    z, log_abs_det = self.bijection.inverse_and_log_det(x, condition)  # condition passed to bijection
    p_z = self.base_dist._log_prob(z, condition)  # condition passed to base dist
    return p_z + log_abs_det
```

The `_vectorize` method handles batched conditions automatically:

```python
# When cond_shape is not None:
in_shapes = [self.shape, self.cond_shape]   # both x and condition have core shapes
out_shapes = [()]                            # output is scalar
# signature: "(d1),(d2)->()" → broadcasts over leading dimensions
```

When `cond_shape is None`, the `condition` argument is excluded from vectorization:
```python
ex = frozenset([1]) if self.cond_shape is None else frozenset()
return jnp.vectorize(_check_shapes(method), signature=signature, excluded=ex)
```

### 4.3 Conditional Distribution Examples

#### Method 1: Conditional Bijection

The primary mechanism is **conditional bijections**. A bijection's `transform_and_log_det(x, condition)` can depend on the condition:

```python
class AdditiveCondition(AbstractBijection):
    """y = x + f(condition), where f is a neural network."""
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]
    module: Callable

    def __init__(self, module, shape, cond_shape):
        self.module = module
        self.shape = shape
        self.cond_shape = cond_shape

    def transform_and_log_det(self, x, condition=None):
        return x + self.module(condition), jnp.zeros(())

    def inverse_and_log_det(self, y, condition=None):
        return y - self.module(condition), jnp.zeros(())
```

#### Method 2: Conditional Autoregressive Flows

The `MaskedAutoregressive` bijection naturally supports conditioning:

```python
class MaskedAutoregressive(AbstractBijection):
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None  # not None when cond_dim is specified

    def __init__(self, key, *, transformer, dim, cond_dim=None, ...):
        if cond_dim is None:
            self.cond_shape = None
        else:
            self.cond_shape = (cond_dim,)
            # conditioning variables get rank -1 (no masking)
            in_ranks = jnp.hstack((jnp.arange(dim), -jnp.ones(cond_dim, int)))
```

#### Creating Conditional Flows

```python
import flowjax
from flowjax.flows import masked_autoregressive_flow
from flowjax.distributions import StandardNormal

# Conditional MAF: p(x | condition) where x ∈ R^3, condition ∈ R^2
base_dist = StandardNormal((3,))
flow = masked_autoregressive_flow(
    key=jr.key(0),
    base_dist=base_dist,
    cond_dim=2,        # makes it conditional
    flow_layers=4,
)

# flow.cond_shape == (2,)
# flow.shape == (3,)

# Usage:
x = jnp.ones(3)
condition = jnp.array([0.5, 1.0])
log_p = flow.log_prob(x, condition=condition)

# Batched:
x_batch = jnp.ones((100, 3))
cond_batch = jnp.ones((100, 2))
log_p_batch = flow.log_prob(x_batch, condition=cond_batch)  # shape (100,)

# Sampling:
samples = flow.sample(jr.key(1), sample_shape=(1000,), condition=condition)
# shape (1000, 3)
```

### 4.4 Could This Be Used for $X|Y \sim \mathcal{N}(\mu + \gamma Y, \Sigma Y)$?

**Yes, with caveats.** The FlowJAX conditional mechanism is designed for arbitrary conditional densities parameterized by neural networks. For the specific case of normal-variance mixtures ($X|Y \sim \mathcal{N}(\mu + \gamma Y, \Sigma Y)$), two approaches are possible:

#### Approach A: Custom Conditional Bijection

```python
class NormalVarianceMixtureBijection(AbstractBijection):
    """y = L_Sigma * sqrt(mixing_val) * z + mu + gamma * mixing_val"""
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]  # shape of mixing variable Y
    mu: Array
    gamma: Array
    L_Sigma: Array  # Cholesky factor

    def transform_and_log_det(self, z, condition=None):
        # z ~ N(0, I), condition = Y (mixing variable)
        y_val = condition  # scalar or vector mixing value
        x = self.mu + self.gamma * y_val + self.L_Sigma @ z * jnp.sqrt(y_val)
        log_det = (0.5 * jnp.log(y_val) * self.shape[0]
                   + jnp.log(jnp.abs(jnp.diag(self.L_Sigma))).sum())
        return x, log_det
```

#### Approach B: Direct Implementation (More Natural for normix)

For a mathematical statistics library, bypassing the bijection framework and implementing `_log_prob` and `_sample` directly is cleaner:

```python
class JointNormalVarianceMixture(AbstractDistribution):
    """X|Y ~ N(mu + gamma*Y, Sigma*Y), where Y ~ mixing distribution."""
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None  # marginal (Y integrated out)

    mu: Array
    gamma: Array
    L_Sigma: Array
    mixing_dist: AbstractDistribution  # e.g., Gamma, InverseGaussian, GIG

    def _log_prob(self, x, condition=None):
        # Compute marginal log prob by integrating over Y
        # (analytically for specific mixing distributions)
        ...

    def _sample(self, key, condition=None):
        k1, k2 = jr.split(key)
        y = self.mixing_dist._sample(k1)
        z = jr.normal(k2, self.shape)
        return self.mu + self.gamma * y + self.L_Sigma @ z * jnp.sqrt(y)
```

**The FlowJAX conditional mechanism is better suited for**:
- Neural-network-parameterized conditionals
- Normalizing flows where you don't know the analytic form
- Amortized inference

**For normix's exponential family distributions**, direct implementation is more appropriate because:
1. We have closed-form log-probabilities
2. We need the exponential family structure (natural params, sufficient stats, etc.)
3. The "conditioning" in $X|Y$ is not an external input — $Y$ is part of the generative model

---

## Design Implications for normix-JAX

### Architecture Recommendation

Based on this analysis, here's how normix could be structured with JAX:

```python
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

class Distribution(eqx.Module):
    """Base class for all distributions."""

    @abstractmethod
    def log_prob(self, x: Array) -> Array: ...

    @abstractmethod
    def sample(self, key, shape=()) -> Array: ...

    @abstractmethod
    def natural_params(self) -> tuple[Array, ...]: ...

    @abstractmethod
    def expectation_params(self) -> tuple[Array, ...]: ...


class ExponentialFamily(Distribution):
    """Base class with exponential family structure."""

    @abstractmethod
    def sufficient_stats(self, x: Array) -> tuple[Array, ...]: ...

    @abstractmethod
    def log_partition(self) -> Array: ...

    @abstractmethod
    def log_base_measure(self, x: Array) -> Array: ...

    def log_prob(self, x: Array) -> Array:
        theta = jnp.concatenate([jnp.ravel(t) for t in self.natural_params()])
        t_x = jnp.concatenate([jnp.ravel(t) for t in self.sufficient_stats(x)])
        return jnp.dot(theta, t_x) - self.log_partition() + self.log_base_measure(x)
```

### Key Design Decisions

1. **Use `eqx.Module`** (not Flax NNX) — immutable, pytree-native, minimal overhead
2. **Named attributes with Greek letters** (matching current normix convention)
3. **`Parameterize` from paramax** for constrained parameters (softplus for positivity, etc.)
4. **`eqx.tree_at`** for functional updates during EM iteration
5. **`eqx.filter_jit`** and **`eqx.filter_grad`** for JIT and differentiation
6. **`jnp.vectorize`** (FlowJAX pattern) for automatic batching of unbatched methods
7. **`ClassVar` for static attributes** like `cond_shape = None`
