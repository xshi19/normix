"""Exercise source drift detection and selective baseline updates in process."""

import hashlib
from pathlib import Path

import pytest

from formal.lean import check_coverage


@pytest.fixture
def coverage_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, str]:
    module = tmp_path / "pilot.py"
    module.write_text(
        "class Gamma:\n"
        "    def var(self):\n"
        "        return self.alpha / self.beta**2\n"
        "\n"
        "    def mean(self):\n"
        "        return self.alpha / self.beta\n",
        encoding="utf-8",
    )
    normalized = "def var(self):\n    return self.alpha / self.beta**2\n"
    digest = "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    source = 'schema_version = 2\nhash_scope = "function-source-sha256"\n'
    for claim_id, status in [
        ("first", "proved"),
        ("second", "tested"),
        ("bridge", "blocked"),
    ]:
        recorded = "PLACEHOLDER:UNREVIEWED" if status == "blocked" else digest
        source += (
            f'\n[[claim]]\nid = "{claim_id}"\nstatus = "{status}"\n'
            'targets = ["pilot.Gamma.var"]\n'
            f'target_hashes = {{ "pilot.Gamma.var" = "{recorded}" }} # retain this comment\n'
            'fallback_hypothesis_test = "tests/test_formal_properties.py::test_gamma_var_formula"\n'
        )
    path = tmp_path / "formal/lean/coverage.toml"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    monkeypatch.setattr(check_coverage, "ROOT", tmp_path)
    return path, module, digest


def test_coverage_baseline() -> None:
    assert check_coverage.main([]) == 0


def test_source_drift_reports_every_claim(
    coverage_map, capsys: pytest.CaptureFixture[str]
) -> None:
    _, module, _ = coverage_map
    module.write_text(
        module.read_text().replace(
            "self.alpha / self.beta**2", "2 * self.alpha / self.beta**2"
        )
    )
    assert check_coverage.main([]) == 1
    output = capsys.readouterr().out
    assert "needs-review: first -> pilot.Gamma.var: hash mismatch" in output
    assert "needs-review: second -> pilot.Gamma.var: hash mismatch" in output
    assert "skipped: bridge -> pilot.Gamma.var: blocked placeholder" in output


def test_unrelated_function_change_is_ignored(coverage_map) -> None:
    _, module, _ = coverage_map
    module.write_text(
        module.read_text().replace("return self.alpha / self.beta\n", "return 0\n")
    )
    assert check_coverage.main([]) == 0


@pytest.mark.parametrize("status", ["proved", "tested"])
def test_active_placeholder_fails(coverage_map, status: str) -> None:
    path, _, digest = coverage_map
    source = path.read_text().replace('status = "proved"', f'status = "{status}"')
    path.write_text(source.replace(digest, "PLACEHOLDER:UNREVIEWED", 1))
    assert check_coverage.main([]) == 1


def test_planned_fallback_fails(coverage_map) -> None:
    path, _, _ = coverage_map
    path.write_text(
        path.read_text().replace(
            'fallback_hypothesis_test = "', 'fallback_hypothesis_test = "PLANNED: ', 1
        )
    )
    assert check_coverage.main([]) == 1


def test_selected_update_preserves_other_claims_and_formatting(coverage_map) -> None:
    path, _, digest = coverage_map
    original = path.read_text()
    pending = original.replace(digest, "PLACEHOLDER:UNREVIEWED")
    path.write_text(pending)
    assert check_coverage.main(["--update", "--claim", "first"]) == 1
    assert path.read_text() == pending.replace("PLACEHOLDER:UNREVIEWED", digest, 1)
    assert (
        check_coverage.main(["--update", "--claim", "first", "--claim", "second"]) == 0
    )
    assert path.read_text() == original


def test_default_update_skips_blocked_and_is_idempotent(coverage_map) -> None:
    path, _, digest = coverage_map
    original = path.read_text()
    path.write_text(original.replace(digest, "PLACEHOLDER:UNREVIEWED"))
    assert check_coverage.main(["--update"]) == 0
    assert path.read_text() == original
    assert check_coverage.main(["--update"]) == 0
    assert path.read_text() == original


def test_unknown_claim_does_not_write(coverage_map) -> None:
    path, _, _ = coverage_map
    original = path.read_bytes()
    assert check_coverage.main(["--update", "--claim", "missing"]) == 1
    assert path.read_bytes() == original


def test_missing_target_fails(coverage_map) -> None:
    _, module, _ = coverage_map
    module.write_text(
        module.read_text().replace("def var(self):", "def renamed(self):")
    )
    assert check_coverage.main([]) == 1
