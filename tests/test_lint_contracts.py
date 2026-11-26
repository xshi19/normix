"""Project lint and numerical source contracts. Historical examples run in shallow clones."""
import ast
import doctest
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _ruff(*args, source=None):
    return subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--no-cache", *args],
        cwd=ROOT, input=source, capture_output=True, text=True,
    )


def test_project_ruff():
    result = _ruff("normix/", "tests/")
    assert result.returncode == 0, result.stdout + result.stderr


# acdad61 replaced the production kve path. SciPy remains an independent test oracle.
@pytest.mark.parametrize("api", ["kv", "kve"])
def test_bessel_import_policy(api):
    old = f"from scipy.special import {api} as bessel\nvalue = bessel(0.5, 1.0)\n"
    new = "from normix.utils.bessel import log_kv\nvalue = log_kv(0.5, 1.0)\n"
    for path, source, banned in [
        ("normix/utils/bessel.py", old, True),
        ("normix/example.py", new, False),
        ("tests/test_bessel_contract.py", old, False),
    ]:
        result = _ruff("--stdin-filename", path, "-", source=source)
        assert result.returncode == int(banned), result.stdout + result.stderr
        if banned:
            assert "TID251" in result.stdout and f"scipy.special.{api}" in result.stdout


def _bind_import(node, names):
    if isinstance(node, ast.Import):
        for alias in node.names:
            root = alias.name.split(".")[0]
            names[alias.asname or root] = alias.name if alias.asname else root
    elif isinstance(node, ast.ImportFrom):
        for alias in node.names:
            module = "." * node.level + (node.module or "")
            names[alias.asname or alias.name] = f"{module}.{alias.name}"


def _name(node, names):
    if isinstance(node, ast.Name):
        return names.get(node.id, "")
    if isinstance(node, ast.Attribute):
        return f"{_name(node.value, names)}.{node.attr}"
    return ""


def _scaled_cdfs(tree):
    def scan(node, names):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.pop(node.name, None)
            names = names.copy()
        elif isinstance(node, ast.Lambda):
            names = names.copy()
        if isinstance(node, ast.arg):
            names.pop(node.arg, None)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.pop(node.id, None)
        _bind_import(node, names)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            calls = {_name(x.func, names) for x in (node.left, node.right) if isinstance(x, ast.Call)}
            if (calls & {"math.exp", "numpy.exp", "jax.numpy.exp"}
                    and calls & {"scipy.stats.norm.cdf", "jax.scipy.stats.norm.cdf"}):
                yield node.lineno
        for child in ast.iter_child_nodes(node):
            yield from scan(child, names)

    return list(scan(tree, {}))


def _doctest_x64(tree):
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node, clean=False)
        if not doc or ">>>" not in doc:
            continue
        names = {}
        enabled = False
        for example in doctest.DocTestParser().get_examples(doc):
            if example.options.get(doctest.SKIP):
                continue
            for stmt in ast.parse(example.source).body:
                if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    _bind_import(stmt, names)
                    modules = ([stmt.module or ""] if isinstance(stmt, ast.ImportFrom)
                               else [alias.name for alias in stmt.names])
                    if not enabled and any(m.split(".")[0] == "normix" for m in modules):
                        findings.append(node.body[0].lineno + example.lineno + stmt.lineno - 1)
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    call = stmt.value
                    if (_name(call.func, names) == "jax.config.update" and len(call.args) == 2
                            and isinstance(call.args[0], ast.Constant)
                            and call.args[0].value == "jax_enable_x64"):
                        enabled = isinstance(call.args[1], ast.Constant) and call.args[1].value is True
    return findings


def _missing_callbacks(tree):
    loops = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name == "_multistart_loop"]
    assert len(loops) == 1, "Locate the multistart retry path in solvers.py"
    solver = "normix.fitting.solvers.solve_bregman"
    names = {"solve_bregman": solver}
    for node in tree.body:
        _bind_import(node, names)
    for node in ast.walk(loops[0]):
        _bind_import(node, names)

    calls = [node for node in ast.walk(loops[0])
             if isinstance(node, ast.Call) and _name(node.func, names) == solver]
    assert calls, "Check derivative forwarding at the new solver entry point"
    findings = []
    for call in calls:
        forwarded = {kw.arg for kw in call.keywords
                     if isinstance(kw.value, ast.Name) and kw.value.id == kw.arg}
        if not {"grad_fn", "hess_fn"} <= forwarded:
            findings.append(call.lineno)
    return findings


def test_production_numerical_contracts():
    findings = []
    for path in sorted((ROOT / "normix").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        relative = path.relative_to(ROOT)
        for line in _scaled_cdfs(tree):
            findings.append(f"{relative}:{line}: use exp(log_scale + log_ndtr(t))")
        for line in _doctest_x64(tree):
            findings.append(f"{relative}:{line}: enable jax_enable_x64 before importing normix")
    path = Path("normix/fitting/solvers.py")
    for line in _missing_callbacks(ast.parse((ROOT / path).read_text())):
        findings.append(f"{path}:{line}: forward grad_fn and hess_fn to solve_bregman")
    assert not findings, "\n".join(findings)


# 02d2a8e repaired the inverse Gaussian CDF when its exponential scale overflows.
@pytest.mark.parametrize(("expression", "bad"), [
    ("jnp.exp(a) * jax.scipy.stats.norm.cdf(t)", True),
    ("phi(t) * jnp.exp(a)", True),
    ("jnp.exp(a + jax.scipy.special.log_ndtr(t))", False),
    ("custom.exp(a) * phi(t)", False),
])
def test_scaled_cdf_examples(expression, bad):
    imports = "import jax\nimport jax.numpy as jnp\nfrom jax.scipy.stats.norm import cdf as phi\n"
    assert bool(_scaled_cdfs(ast.parse(imports + expression))) == bad


def test_unrelated_cdf_in_local_scope():
    source = """import numpy as np
from scipy.stats import norm
def scaled(norm):
    return np.exp(a) * norm.cdf(t)
"""
    assert not _scaled_cdfs(ast.parse(source))


# 8bfcaa1 made the Bessel symmetry example meet its 1e-10 tolerance in isolation.
@pytest.mark.parametrize(("setup", "late", "bad"), [
    ("", "", True),
    (">>> jax.config.update('jax_enable_x64', True)", "", False),
    (">>> jax.config.update('jax_enable_x64', False)", "", True),
    ("Enable jax_enable_x64 first.", ">>> jax.config.update('jax_enable_x64', True)", True),
])
def test_doctest_precision_examples(setup, late, bad):
    doc = f">>> import jax\n{setup}\n>>> from normix import log_kv\n{late}"
    assert bool(_doctest_x64(ast.parse(repr(doc)))) == bad


# e51c1a7 omitted both callbacks on the final retry. 7c5deac (#173) restored them.
@pytest.mark.parametrize(("retry", "bad"), [
    ("solve_bregman(f, eta, starts[0])", True),
    ("solvers.solve_bregman(f, eta, starts[0], grad_fn=grad_fn, hess_fn=None)", True),
    ("solve_bregman(f, eta, starts[0], grad_fn=grad_fn, hess_fn=hess_fn)", False),
])
def test_multistart_callback_examples(retry, bad):
    source = f"""from normix.fitting import solvers
def _multistart_loop(f, eta, starts, *, grad_fn, hess_fn):
    for t0 in starts:
        solve_bregman(f, eta, t0, grad_fn=grad_fn, hess_fn=hess_fn)
    return {retry}
"""
    assert bool(_missing_callbacks(ast.parse(source))) == bad
