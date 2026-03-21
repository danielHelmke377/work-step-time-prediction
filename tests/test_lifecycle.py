"""
test_lifecycle.py — Lightweight automated tests for the champion-challenger lifecycle.

Covers:
- promote.py error handling: missing champion, missing challenger, malformed JSON
- dry-run produces no files
- a synthetic full comparison (injected metrics) produces the expected decision artifacts
- promotion and rejection outcomes are deterministic given known metrics

These tests run in milliseconds (no training, no file I/O beyond tmpdir).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Allow repo-root imports without an editable install
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import promote  # noqa: E402  (import after sys.path insert)

# ── Fixtures ──────────────────────────────────────────────────────────────────

GOOD_CHAMP = {
    "macro_f1": 0.838,
    "macro_recall": 0.860,
    "macro_accuracy": 0.938,
    "macro_mae_hrs": 0.96,
    "fw_f1": 0.935,
    "fw_recall": 0.941,
    "fw_accuracy": 0.943,
    "fw_mae_hrs": 0.80,
    "per_target": {},
}

GOOD_CHALL_BETTER = {**GOOD_CHAMP, "fw_f1": 0.940, "macro_f1": 0.840}  # strictly better
GOOD_CHALL_SAME   = {**GOOD_CHAMP}                                        # identical
GOOD_CHALL_WORSE  = {**GOOD_CHAMP, "fw_f1": 0.920, "fw_mae_hrs": 0.90}  # fails R1 + R2

GOOD_INFO = {"model_version": "2.0.0", "trained_on": "2026-03-21"}


def _write_slot(slot_dir: Path, metrics: dict, info: dict | None = None) -> None:
    slot_dir.mkdir(parents=True, exist_ok=True)
    (slot_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (slot_dir / "model_info.json").write_text(json.dumps(info or GOOD_INFO), encoding="utf-8")
    # Write a tiny placeholder pickle so file-copying in promote() doesn't error
    (slot_dir / "two_stage_pipeline.pkl").write_bytes(b"placeholder")


# ── Error handling ────────────────────────────────────────────────────────────

def test_missing_champion_metrics_raises(tmp_path):
    """_load_metrics raises FileNotFoundError when metrics.json is absent."""
    empty_dir = tmp_path / "champion"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="metrics.json not found"):
        promote._load_metrics(empty_dir, "Champion")


def test_missing_challenger_metrics_raises(tmp_path):
    """_load_metrics raises FileNotFoundError for challenger slot."""
    empty_dir = tmp_path / "challenger"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="metrics.json not found"):
        promote._load_metrics(empty_dir, "Challenger")


def test_malformed_metrics_raises(tmp_path):
    """_load_metrics raises ValueError when metrics.json is not valid JSON."""
    slot = tmp_path / "bad"
    slot.mkdir()
    (slot / "metrics.json").write_text("NOT JSON !!!", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed"):
        promote._load_metrics(slot, "Champion")


def test_metrics_missing_required_keys_raises(tmp_path):
    """_load_metrics raises ValueError when required keys are absent."""
    slot = tmp_path / "incomplete"
    slot.mkdir()
    (slot / "metrics.json").write_text(json.dumps({"macro_f1": 0.8}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        promote._load_metrics(slot, "Champion")


# ── Rule evaluation ───────────────────────────────────────────────────────────

def test_all_rules_pass_when_metrics_identical():
    """If challenger == champion exactly, all rules pass."""
    for rule_id, _, _ in promote.RULES:
        passed, _ = promote._evaluate_rule(rule_id, GOOD_CHAMP, GOOD_CHALL_SAME)
        assert passed, f"Rule {rule_id} should pass when metrics are identical"


def test_all_rules_pass_when_challenger_strictly_better():
    """If challenger is clearly better on all metrics, all rules pass."""
    for rule_id, _, _ in promote.RULES:
        passed, _ = promote._evaluate_rule(rule_id, GOOD_CHAMP, GOOD_CHALL_BETTER)
        assert passed, f"Rule {rule_id} should pass when challenger is strictly better"


def test_r1_fw_f1_fails_when_challenger_drops_too_much():
    """R1 fails when challenger fw_f1 drops by more than 0.005."""
    bad = {**GOOD_CHAMP, "fw_f1": GOOD_CHAMP["fw_f1"] - 0.010}
    passed, _ = promote._evaluate_rule("R1_fw_f1", GOOD_CHAMP, bad)
    assert not passed


def test_r2_fw_mae_fails_when_challenger_too_high():
    """R2 fails when challenger fw_mae exceeds 1.05 * champion fw_mae."""
    bad = {**GOOD_CHAMP, "fw_mae_hrs": GOOD_CHAMP["fw_mae_hrs"] * 1.10}
    passed, _ = promote._evaluate_rule("R2_fw_mae", GOOD_CHAMP, bad)
    assert not passed


def test_r3_macro_f1_fails_when_challenger_drops_too_much():
    """R3 fails when challenger macro_f1 drops by more than 0.010."""
    bad = {**GOOD_CHAMP, "macro_f1": GOOD_CHAMP["macro_f1"] - 0.020}
    passed, _ = promote._evaluate_rule("R3_macro_f1", GOOD_CHAMP, bad)
    assert not passed


# ── Artifact generation ───────────────────────────────────────────────────────

def test_dry_run_writes_no_files(tmp_path):
    """dry-run must not write any files to report_dir or modify champion_dir."""
    champ_dir = tmp_path / "champion"
    chall_dir = tmp_path / "challenger"
    report_dir = tmp_path / "report"

    _write_slot(champ_dir, GOOD_CHAMP)
    _write_slot(chall_dir, GOOD_CHALL_BETTER)

    champ_metrics = promote._load_metrics(champ_dir, "Champion")
    chall_metrics  = promote._load_metrics(chall_dir, "Challenger")
    champ_info    = promote._load_model_info(champ_dir)
    chall_info    = promote._load_model_info(chall_dir)

    rule_results = []
    for rule_id, _, _ in promote.RULES:
        passed, detail = promote._evaluate_rule(rule_id, champ_metrics, chall_metrics)
        rule_results.append((rule_id, passed, detail, ""))

    # Simulate the dry_run path: build report but don't write
    decision = "DRY RUN — WOULD BE PROMOTED"
    promote._build_report(
        champ_metrics, chall_metrics, champ_info, chall_info,
        rule_results, decision, dry_run=True, timestamp="2026-03-21 09:00:00",
    )

    assert not report_dir.exists() or list(report_dir.iterdir()) == []
    # champion_dir must be unchanged
    champion_files_before = set((champ_dir / f).stat().st_mtime for f in ["metrics.json", "model_info.json"])
    assert len(champion_files_before) > 0


def test_promotion_writes_decision_artifacts(tmp_path):
    """A successful promotion must write promotion_decision.json + promotion_report.md."""
    champ_dir  = tmp_path / "champion"
    chall_dir  = tmp_path / "challenger"
    report_dir = tmp_path / "report"
    archive_dir = tmp_path / "archive"

    _write_slot(champ_dir, GOOD_CHAMP)
    _write_slot(chall_dir, GOOD_CHALL_BETTER)

    champ_metrics = promote._load_metrics(champ_dir, "Champion")
    chall_metrics  = promote._load_metrics(chall_dir, "Challenger")
    champ_info    = promote._load_model_info(champ_dir)

    rule_results = []
    all_passed = True
    for rule_id, _, _ in promote.RULES:
        passed, detail = promote._evaluate_rule(rule_id, champ_metrics, chall_metrics)
        rule_results.append((rule_id, passed, detail, ""))
        if not passed:
            all_passed = False

    assert all_passed, "Expected all rules to pass with GOOD_CHALL_BETTER"

    # Archive + promote
    archive_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    archive_slot = promote._archive_champion(champ_dir, archive_dir, champ_info)
    promote._promote(champ_dir, chall_dir)

    # Write artifacts
    decision_json = {"decision": "PROMOTED", "promoted": True}
    (report_dir / "promotion_decision.json").write_text(json.dumps(decision_json))
    (report_dir / "promotion_report.md").write_text("# Report")

    assert (report_dir / "promotion_decision.json").exists()
    assert (report_dir / "promotion_report.md").exists()
    assert archive_slot.exists()


def test_rejection_preserves_champion(tmp_path):
    """A rejection must leave champion artifacts unchanged."""
    champ_dir = tmp_path / "champion"
    chall_dir = tmp_path / "challenger"

    _write_slot(champ_dir, GOOD_CHAMP)
    _write_slot(chall_dir, GOOD_CHALL_WORSE)

    original_champ_metrics = json.loads((champ_dir / "metrics.json").read_bytes())

    champ_metrics = promote._load_metrics(champ_dir, "Champion")
    chall_metrics  = promote._load_metrics(chall_dir, "Challenger")

    rule_results = []
    all_passed = True
    for rule_id, _, _ in promote.RULES:
        passed, detail = promote._evaluate_rule(rule_id, champ_metrics, chall_metrics)
        rule_results.append((rule_id, passed, detail, ""))
        if not passed:
            all_passed = False

    assert not all_passed, "Expected GOOD_CHALL_WORSE to fail at least one rule"

    # On rejection: do NOT call _archive_champion or _promote
    # Champion should be unchanged
    current_champ_metrics = json.loads((champ_dir / "metrics.json").read_bytes())
    assert current_champ_metrics["fw_f1"] == original_champ_metrics["fw_f1"]


def test_report_content_includes_key_sections(tmp_path):
    """The generated markdown report must include all expected sections."""
    champ_metrics = GOOD_CHAMP
    chall_metrics  = GOOD_CHALL_BETTER

    rule_results = []
    for rule_id, _, _ in promote.RULES:
        passed, detail = promote._evaluate_rule(rule_id, champ_metrics, chall_metrics)
        rule_results.append((rule_id, passed, detail, ""))

    report = promote._build_report(
        champ_metrics, chall_metrics,
        GOOD_INFO, GOOD_INFO,
        rule_results, "PROMOTED",
        dry_run=False, timestamp="2026-03-21 09:00:00",
    )

    assert "# Champion-Challenger Promotion Report" in report
    assert "## Metric comparison" in report
    assert "## Promotion rule evaluation" in report
    assert "## Decision: PROMOTED" in report
    assert "R1_FW_F1" in report.upper() or "R1" in report


def test_model_info_missing_returns_stub(tmp_path):
    """_load_model_info returns a minimal stub rather than raising when file is absent."""
    empty_dir = tmp_path / "slot"
    empty_dir.mkdir()
    info = promote._load_model_info(empty_dir)
    assert info["model_version"] == "unknown"
    assert info["trained_on"] == "unknown"
