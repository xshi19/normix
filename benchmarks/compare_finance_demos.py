"""Execute finance MyST cells and save numeric results for before/after audits.

Example, from the repository root::

    uv run python benchmarks/compare_finance_demos.py --output /tmp/demos-after.json

Use ``--tutorial-dir`` and ``--module-root`` to run a saved baseline source
against a saved package. The data working directory stays in this checkout.
Each page executes in a fresh process. Figures are closed, but every code
cell runs. No stored notebook outputs or execution caches are read.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def _worker(page: Path, module_root: Path, output: Path) -> None:
    sys.path.insert(0, str(module_root))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import numpy as np
    import matplotlib.pyplot as plt
    root = Path(__file__).resolve().parents[1]
    os.chdir(root / "docs/tutorials/finance")
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(re.findall(r"```\{code-cell\} python\n(.*?)\n```", page.read_text(), re.S)):
        exec(compile(cell, f"{page}:cell{index}", "exec"), namespace)
        plt.close("all")
    keys = {
        "04": ("cvar_a", "cvar_brute", "var_a", "path", "w_opt", "g_analytic", "gw_analytic"),
        "05": ("summary", "weights"),
        "06": ("w0",),
        "07": ("table", "portfolios"),
    }[page.name[:2]]
    def convert(value):
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if hasattr(value, "to_dict"):
            return convert(value.to_dict())
        return np.asarray(value).tolist()
    data = {k: convert(namespace[k]) for k in keys}
    if page.name.startswith("05"):
        data["frontiers"] = {k: {"risk": convert(v.risk), "weights": convert(v.weights)}
                             for k, v in namespace["frontiers"].items()}
        data["surface"] = convert(namespace["surface"].risk)
    if page.name.startswith("06"):
        result, tc, Q = namespace["result"], namespace["tc"], namespace.get("Q", namespace.get("Y"))
        data.update(weights=convert(result.weights), turnover=convert(result.turnover),
                    objective=convert(tc.true_objective_at(result.weights, namespace["w0"], Q)))
    output.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tutorial-dir", type=Path, default=root / "docs/tutorials/finance")
    parser.add_argument("--module-root", type=Path, default=root)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page", type=Path)
    args = parser.parse_args()
    if args.page:
        _worker(args.page.resolve(), args.module_root.resolve(), args.output.resolve())
        return
    data = {}
    for prefix in ("04", "05", "06", "07"):
        page = next(args.tutorial_dir.glob(prefix + "_*.md"))
        out = args.output.with_name(args.output.stem + "-" + prefix + ".json")
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--page", str(page.resolve()),
                        "--module-root", str(args.module_root.resolve()), "--output", str(out.resolve())], check=True)
        data[page.stem] = json.loads(out.read_text())
    args.output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
