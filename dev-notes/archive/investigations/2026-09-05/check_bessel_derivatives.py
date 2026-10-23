#!/usr/bin/env python3
"""Independent numerical checks; does not modify Normix.
Run: python check_bessel_derivatives.py --repo /path/to/normix --output results.json
Dependencies: normix and its dependencies, mpmath. Enable float64 before import.
The original audit scripts are supplied separately; this is an enhanced check.
"""
import argparse
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=pathlib.Path)
parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('bessel-derivative-results.json'))
parser.add_argument('--dps', type=int, default=60)
args = parser.parse_args()
if args.dps < 40:
    parser.error('--dps must be at least 40')
if args.repo:
    sys.path.insert(0, str(args.repo.resolve()))
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import mpmath as mp
import numpy as np
from normix.utils.bessel import log_kv, _hankel_log_kv


def differentiated_reference(v, z, dps):
    with mp.workdps(dps):
        # Convert identical binary64 input values, avoiding a decimal/binary mismatch.
        nu, arg = mp.mpf(v), mp.mpf(z)
        f = lambda order: mp.log(mp.besselk(order, arg))
        return [f(nu), mp.diff(f, nu, 1), mp.diff(f, nu, 2)]


def integral_reference(v, z, dps):
    """No besselk or diff calls: integrate centered whole-line density moments.

    Tail truncation is numerical, not a certified interval bound. Increasing
    precision also increases the requested tail log-drop. Agreement with the
    differentiated reference is checked explicitly by the caller.
    """
    with mp.workdps(dps):
        nu, arg = mp.mpf(v), mp.mpf(z)
        mode = mp.asinh(nu / arg)
        kappa = mp.sqrt(nu * nu + arg * arg)
        scale = mp.sqrt(kappa)
        def log_weight(y):
            x = y / scale
            return nu * (x - mp.sinh(x)) - kappa * (mp.cosh(x) - 1)
        target = (dps + 15) * mp.log(10)
        left, right = mp.mpf(-1), mp.mpf(1)
        while log_weight(left) > -target:
            left *= 2
        while log_weight(right) > -target:
            right *= 2
        nodes = sorted(set([left, right, mp.mpf(0)] +
                           [mp.mpf(x) for x in (-16, -8, -4, -2, -1, 1, 2, 4, 8, 16)
                            if left < x < right]))
        weight = lambda y: mp.exp(log_weight(y))
        normalizer = mp.quad(weight, nodes)
        mean_y = mp.quad(lambda y: y * weight(y), nodes) / normalizer
        variance_y = mp.quad(lambda y: (y - mean_y)**2 * weight(y), nodes) / normalizer
        value = nu * mode - kappa + mp.log(normalizer / scale) - mp.log(2)
        return [value, mode + mean_y / scale, variance_y / kappa]


points = np.array([(0., 1.), (.5, 1.), (25., .1), (25., 1.),
                   (25., 1e-6), (1., 1e4), (5., 1e6)])
value_fn = jax.jit(jax.vmap(log_kv))
first_fn = jax.jit(jax.vmap(jax.grad(log_kv, argnums=0)))
second_fn = jax.jit(jax.vmap(jax.grad(jax.grad(log_kv, argnums=0), argnums=0)))
values = np.asarray(value_fn(points[:, 0], points[:, 1]))
first = np.asarray(first_fn(points[:, 0], points[:, 1]))
second = np.asarray(second_fn(points[:, 0], points[:, 1]))
rows = []
for i, (v, z) in enumerate(points):
    low = differentiated_reference(float(v), float(z), args.dps)
    high = differentiated_reference(float(v), float(z), args.dps + 30)
    integral = integral_reference(float(v), float(z), args.dps)
    with mp.workdps(args.dps + 30):
        precision_error = max(abs(a-b) / max(1, abs(b)) for a,b in zip(low, high))
        integral_error = max(abs(a-b) / max(1, abs(b)) for a,b in zip(integral, high))
        tolerance = mp.mpf(10) ** (-min(30, args.dps // 2))
        if max(precision_error, integral_error) > tolerance:
            raise RuntimeError(f'Reference disagreement at {(v,z)}: {precision_error}, {integral_error}')
        reference = [mp.nstr(x, args.dps) for x in high]
    h = 1e-5
    # This direct stencil differs from nested AD through Normix's custom JVP.
    direct_second = float((log_kv(v+h,z)-2*log_kv(v,z)+log_kv(v-h,z))/h**2)
    row = dict(v=float(v), z=float(z), reference=dict(zip(['log_k','d_order','d2_order'], reference)),
               normix=dict(log_k=float(values[i]), d_order=float(first[i]),
                           d2_order_nested_ad=float(second[i]), d2_order_direct_fd=direct_second),
               precision_check_scaled_error=float(precision_error),
               integral_check_scaled_error=float(integral_error))
    rows.append(row)
    print(json.dumps(row), flush=True)

# A narrow diagnostic: AD through a smooth internal Hankel kernel, with no
# custom order finite difference. This is not a replacement implementation.
hankel_d2 = jax.jit(jax.grad(jax.grad(_hankel_log_kv, argnums=0), argnums=0))
hankel = [dict(v=v, z=z, direct_kernel_ad_d2=float(hankel_d2(jnp.asarray(v),jnp.asarray(z))))
          for v,z in [(1.,1e4),(5.,1e6)]]
commit = None
if args.repo:
    result = subprocess.run(['git','-C',str(args.repo),'rev-parse','HEAD'],capture_output=True,text=True)
    if result.returncode == 0:
        commit = result.stdout.strip()
versions = {p: importlib.metadata.version(p) for p in ('jax','jaxlib','numpy','mpmath','scipy')}
payload = dict(python=platform.python_version(), versions=versions, normix_commit=commit,
               reference_dps=[args.dps,args.dps+30], cases=rows, hankel_diagnostic=hankel,
               note='Selected-point audit; agreement is not a formal error certificate or domain-wide guarantee.')
args.output.write_text(json.dumps(payload,indent=2)+'\n')
print('Hankel direct-kernel AD:', hankel)
print('Saved', args.output)
