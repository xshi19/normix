"""S9 guard: ``backend='jax'`` must not hide a host callback.

The token scan covers ``normix`` sources. The jaxpr scan covers the traced
GIG solve, including nested ``while``, ``cond``, ``scan``, ``jit`` /
``pjit``, and ``custom_jvp`` bodies.

See ``dev-notes/design/solvers_and_bessel.md`` § 4.1.
"""

import tokenize
from pathlib import Path

import jax
import jax.extend.core as jcore
import jax.numpy as jnp
import pytest

from normix import GIG

PACKAGE = Path(__file__).resolve().parents[1] / "normix"
FORBIDDEN = {"pure_callback", "io_callback"}


def _hits(path):
    with path.open("rb") as f:
        for tok in tokenize.tokenize(f.readline):
            if tok.type == tokenize.NAME and tok.string in FORBIDDEN:
                yield f"{path.relative_to(PACKAGE.parent)}:{tok.start[0]}: {tok.string}"


def test_no_host_callbacks_in_package():
    hits = [h for p in sorted(PACKAGE.rglob("*.py")) for h in _hits(p)]
    assert not hits, (
        "Host callbacks violate S9 (solvers_and_bessel.md § 4.1):\n" + "\n".join(hits)
    )


def _primitive_names(jaxpr):
    """Return primitive names in ``jaxpr`` and nested sub-jaxprs."""
    names = []
    seen = set()

    def walk(node):
        inner = node.jaxpr if isinstance(node, jcore.ClosedJaxpr) else node
        if id(inner) in seen:
            return
        seen.add(id(inner))
        for eqn in inner.eqns:
            names.append(eqn.primitive.name)
            for value in eqn.params.values():
                for sub in _jaxprs_in(value):
                    walk(sub)

    walk(jaxpr)
    return names


def _jaxprs_in(value):
    if isinstance(value, (jcore.Jaxpr, jcore.ClosedJaxpr)):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _jaxprs_in(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _jaxprs_in(item)


@pytest.mark.contract
def test_jaxpr_walker_enters_callback_bodies():
    """Nested while, cond, scan, jit, and custom_jvp bodies are scanned."""

    def host(y):
        return y + 1.0

    def call(x):
        return jax.pure_callback(host, x, x)

    @jax.custom_jvp
    def wrapped(x):
        return call(x)

    @wrapped.defjvp
    def wrapped_jvp(primals, tangents):
        x, dx = primals[0], tangents[0]
        return x, dx

    def step(v):
        v = jax.lax.cond(v[0] >= 0.0, call, lambda z: z, v)
        v, _ = jax.lax.scan(lambda c, _: (call(c), c), v, None, length=1)
        return v

    def fn(x):
        looped = jax.lax.while_loop(lambda v: v[0] < 1.0, step, x)
        return wrapped(jax.jit(call)(looped))

    names = _primitive_names(jax.make_jaxpr(fn)(jnp.zeros(1)))
    assert names.count("pure_callback") == 4
    for primitive in ("while", "cond", "scan", "custom_jvp_call"):
        assert primitive in names
    assert "jit" in names or "pjit" in names


@pytest.mark.contract
def test_gig_jax_solve_jaxpr_has_no_callback():
    """``GIG.from_expectation(backend='jax')`` jaxpr has no callback primitive."""
    gig = GIG(0.5, 1.0, 1.0)
    eta = gig.expectation_params()
    theta0 = gig.natural_params()

    def cold(target):
        return GIG.from_expectation(
            target, backend="jax", method="newton",
        ).natural_params()

    def warm(target, start):
        return GIG.from_expectation(
            target, theta0=start, backend="jax", method="newton",
        ).natural_params()

    batch = jnp.stack((eta, eta))
    starts = jnp.stack((theta0, theta0))
    specs = (
        ("cold", cold, (eta,)),
        ("warm", warm, (eta, theta0)),
        ("vmap-cold", jax.vmap(cold), (batch,)),
        ("vmap-warm", jax.vmap(warm), (batch, starts)),
    )
    with jax.default_device(jax.devices("cpu")[0]):
        for label, fn, args in specs:
            names = _primitive_names(jax.make_jaxpr(fn)(*args))
            hits = sorted({name for name in names if "callback" in name})
            assert not hits, f"{label}: {hits}"
            assert "while" in names, f"{label}: solver loop was not in the jaxpr"
