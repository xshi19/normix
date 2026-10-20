"""S9 guard: ``backend='jax'`` must not hide a host callback.

See ``dev-notes/design/solvers_and_bessel.md`` § 4.1.
"""

import tokenize
from pathlib import Path

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
