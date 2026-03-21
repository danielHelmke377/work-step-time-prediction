"""
promote.py — Champion-Challenger promotion workflow.

This script implements a lightweight, file-based model lifecycle for the
work-step-time-prediction repository.  It reads pre-computed evaluation
metrics from a trained challenger and the live champion, applies explicit
promotion rules, writes a machine-readable decision and a human-readable
report, and optionally promotes the challenger into the champion slot.

Lifecycle structure
-------------------
  models/
  ├── two_stage_pipeline.pkl   ← live champion (API loads this)
  ├── model_info.json          ← champion metadata
  ├── metrics.json             ← champion evaluation metrics
  ├── challenger/              ← challenger slot (train here first)
  │   ├── two_stage_pipeline.pkl
  │   ├── model_info.json
  │   └── metrics.json
  └── archive/                 ← retired champions (auto-managed)
      └── champion_v2.0.0_20260321_120000/
          ├── two_stage_pipeline.pkl
          ├── model_info.json
          └── metrics.json

Typical workflow
----------------
  # 1. Train a challenger (does not touch the live champion)
  python scripts/train.py --target-dir models/challenger

  # 2. Inspect the comparison without promoting
  python scripts/promote.py --dry-run

  # 3. Promote if satisfied
  python scripts/promote.py

  # Or use the Makefile shortcuts:
  make train-challenger
  make promote-dry
  make promote

Promotion rules (all three must pass)
--------------------------------------
  R1  FW-F1     challenger fw_f1  >= champion fw_f1  - 0.005
  R2  FW-MAE    challenger fw_mae <= champion fw_mae  * 1.05
  R3  Macro-F1  challenger macro_f1 >= champion macro_f1 - 0.010
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Promotion rules ───────────────────────────────────────────────────────────

RULES: list[tuple[str, str, str]] = [
    # (rule_id, description, explanation)
    (
        "R1_fw_f1",
        "Frequency-weighted F1 must not drop by more than 0.005",
        "Primary business metric. A ≤0.5pp dip is tolerated "
        "(noise from stochastic training); any larger drop means the "
        "challenger is materially worse on common work steps.",
    ),
    (
        "R2_fw_mae",
        "Frequency-weighted MAE must not increase by more than 5%",
        "Duration quality guard. Body-shop planning depends on accurate "
        "hour estimates; a challenger that inflates MAE by more than 5% "
        "would degrade scheduling quality.",
    ),
    (
        "R3_macro_f1",
        "Macro F1 must not drop by more than 0.010",
        "Rare-step fairness guard. Frequency-weighting can hide collapse "
        "on low-frequency steps. This rule ensures no individual target "
        "degrades catastrophically.",
    ),
]


def _evaluate_rule(rule_id: str, champ: dict, chall: dict) -> tuple[bool, str]:
    """Return (passed, detail_string) for one rule."""
    if rule_id == "R1_fw_f1":
        threshold = champ["fw_f1"] - 0.005
        actual = chall["fw_f1"]
        passed = actual >= threshold
        detail = (
            f"challenger fw_f1={actual:.4f}  "
            f"vs  champion fw_f1={champ['fw_f1']:.4f}  "
            f"(min required: {threshold:.4f})"
        )
    elif rule_id == "R2_fw_mae":
        threshold = champ["fw_mae_hrs"] * 1.05
        actual = chall["fw_mae_hrs"]
        passed = actual <= threshold
        detail = (
            f"challenger fw_mae={actual:.4f} hrs  "
            f"vs  champion fw_mae={champ['fw_mae_hrs']:.4f} hrs  "
            f"(max allowed: {threshold:.4f})"
        )
    elif rule_id == "R3_macro_f1":
        threshold = champ["macro_f1"] - 0.010
        actual = chall["macro_f1"]
        passed = actual >= threshold
        detail = (
            f"challenger macro_f1={actual:.4f}  "
            f"vs  champion macro_f1={champ['macro_f1']:.4f}  "
            f"(min required: {threshold:.4f})"
        )
    else:
        raise ValueError(f"Unknown rule: {rule_id}")

    return passed, detail


# ── Metric loading ────────────────────────────────────────────────────────────

def _load_metrics(directory: Path, label: str) -> dict:
    """Load metrics.json from directory, with clear errors."""
    path = directory / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{label} metrics.json not found at {path}.\n"
            f"Train the {label.lower()} first:\n"
            f"  {'make train' if label == 'Champion' else 'make train-challenger'}"
        )
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed {label} metrics.json at {path}: {exc}") from exc

    required = {"macro_f1", "fw_f1", "fw_mae_hrs", "macro_mae_hrs"}
    missing = required - set(data)
    if missing:
        raise ValueError(
            f"{label} metrics.json is missing keys: {sorted(missing)}\n"
            f"  Re-train to regenerate a valid metrics.json."
        )
    return data


def _load_model_info(directory: Path) -> dict:
    """Load model_info.json; return a minimal stub if missing."""
    path = directory / "model_info.json"
    if not path.exists():
        return {"model_version": "unknown", "trained_on": "unknown"}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {"model_version": "unknown", "trained_on": "unknown"}


# ── Report generation ─────────────────────────────────────────────────────────

def _build_report(
    champ_metrics: dict,
    chall_metrics: dict,
    champ_info: dict,
    chall_info: dict,
    rule_results: list[tuple[str, bool, str, str]],
    decision: str,
    dry_run: bool,
    timestamp: str,
) -> str:
    """Return a markdown promotion report string."""
    lines = [
        "# Champion-Challenger Promotion Report",
        "",
        f"**Date:** {timestamp}  ",
        f"**Decision:** **{decision}**  ",
        f"**Mode:** {'DRY RUN — no files changed' if dry_run else 'LIVE RUN'}",
        "",
        "---",
        "",
        "## Model versions",
        "",
        "| Slot | Version | Trained on |",
        "|---|---|---|",
        f"| Champion | {champ_info.get('model_version', '?')} | {champ_info.get('trained_on', '?')} |",
        f"| Challenger | {chall_info.get('model_version', '?')} | {chall_info.get('trained_on', '?')} |",
        "",
        "---",
        "",
        "## Metric comparison",
        "",
        "| Metric | Champion | Challenger | Delta |",
        "|---|---:|---:|---:|",
    ]

    metrics_to_show = [
        ("Macro F1",         "macro_f1",       ".4f", False),
        ("Macro MAE (hrs)",  "macro_mae_hrs",  ".4f", True),
        ("Freq-weighted F1", "fw_f1",          ".4f", False),
        ("Freq-weighted MAE (hrs)", "fw_mae_hrs", ".4f", True),
        ("Freq-weighted Accuracy",  "fw_accuracy",  ".4f", False),
    ]

    for label, key, fmt, lower_is_better in metrics_to_show:
        c = champ_metrics.get(key, float("nan"))
        h = chall_metrics.get(key, float("nan"))
        try:
            delta = h - c
            delta_str = f"{delta:+.4f}"
            if lower_is_better:
                trend = "+" if delta < 0 else ("-" if delta > 0 else "=")
            else:
                trend = "+" if delta > 0 else ("-" if delta < 0 else "=")
            arrow = {"+" : "▲", "-": "▼", "=": "="} [trend]
        except TypeError:
            delta_str = "?"
            arrow = "?"
        lines.append(f"| {label} | {c:{fmt}} | {h:{fmt}} | {arrow} {delta_str} |")

    lines += [
        "",
        "---",
        "",
        "## Promotion rule evaluation",
        "",
        "| Rule | Description | Result | Detail |",
        "|---|---|---|---|",
    ]

    for rule_id, passed, detail, _ in rule_results:
        pretty_id = rule_id.upper().replace("_", " ")
        result_str = "PASS" if passed else "FAIL"
        lines.append(f"| `{pretty_id}` | See below | **{result_str}** | {detail} |")

    lines += ["", "### Rule definitions", ""]
    for rule_id, desc, explanation in RULES:
        pretty_id = rule_id.upper().replace("_", " ")
        lines.append(f"**`{pretty_id}`** — {desc}  ")
        lines.append(f"{explanation}")
        lines.append("")

    lines += [
        "---",
        "",
        f"## Decision: {decision}",
        "",
    ]

    if decision == "PROMOTED":
        lines += [
            "All three promotion rules passed. The challenger has been copied into",
            "the live champion slot. The previous champion has been archived under",
            "`models/archive/`.",
        ]
    elif decision == "REJECTED":
        failed = [r[0] for r in rule_results if not r[1]]
        lines += [
            f"**{len(failed)} rule(s) failed:** " + ", ".join(f"`{r}`" for r in failed) + ".",
            "The champion artifact remains unchanged.",
            "The challenger remains in `models/challenger/` for inspection.",
        ]
    elif decision == "DRY RUN — WOULD BE PROMOTED":
        lines += [
            "All rules passed in dry-run mode. No files were changed.",
            "Re-run without `--dry-run` to perform the actual promotion.",
        ]
    elif decision == "DRY RUN — WOULD BE REJECTED":
        failed = [r[0] for r in rule_results if not r[1]]
        lines += [
            f"**{len(failed)} rule(s) failed:** " + ", ".join(f"`{r}`" for r in failed) + ".",
            "No files changed (dry-run mode).",
        ]

    lines.append("")
    return "\n".join(lines)


# ── Promotion action ──────────────────────────────────────────────────────────

_ARTIFACT_FILES = [
    "two_stage_pipeline.pkl",
    "model_info.json",
    "metrics.json",
]


def _archive_champion(champion_dir: Path, archive_dir: Path, champ_info: dict) -> Path:
    """Copy current champion artifacts to a timestamped archive directory."""
    version = champ_info.get("model_version", "unknown").replace("/", "-")
    trained_on = champ_info.get("trained_on", "unknown").replace("-", "")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slot_name = f"champion_v{version}_{trained_on}_{ts}"
    dest = archive_dir / slot_name
    dest.mkdir(parents=True, exist_ok=True)

    for fname in _ARTIFACT_FILES:
        src = champion_dir / fname
        if src.exists():
            shutil.copy2(src, dest / fname)

    return dest


def _promote(champion_dir: Path, challenger_dir: Path) -> None:
    """Overwrite champion slot artifacts with challenger artifacts."""
    for fname in _ARTIFACT_FILES:
        src = challenger_dir / fname
        if src.exists():
            shutil.copy2(src, champion_dir / fname)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    repo_root = _REPO_ROOT
    parser = argparse.ArgumentParser(
        description="Compare a challenger model to the champion and optionally promote it.",
    )
    parser.add_argument(
        "--champion-dir", default=str(repo_root / "models"),
        help="Directory containing champion artifacts (default: models/).",
    )
    parser.add_argument(
        "--challenger-dir", default=str(repo_root / "models" / "challenger"),
        help="Directory containing challenger artifacts (default: models/challenger/).",
    )
    parser.add_argument(
        "--archive-dir", default=str(repo_root / "models" / "archive"),
        help="Directory where retired champions are stored (default: models/archive/).",
    )
    parser.add_argument(
        "--report-dir", default=str(repo_root / "models"),
        help="Directory to write decision artifacts (default: models/).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Evaluate promotion rules and print the report without changing any files.",
    )
    args = parser.parse_args()

    champion_dir  = Path(args.champion_dir)
    challenger_dir = Path(args.challenger_dir)
    archive_dir   = Path(args.archive_dir)
    report_dir    = Path(args.report_dir)
    dry_run       = args.dry_run
    timestamp     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*60}")
    print("  Champion-Challenger Promotion Evaluation")
    print(f"  {'[DRY RUN]' if dry_run else '[LIVE RUN]'}")
    print(f"{'='*60}\n")

    # ── Load metrics ─────────────────────────────────────────────────────────
    try:
        champ_metrics = _load_metrics(champion_dir, "Champion")
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "\nNo champion metrics found. If this is the first lifecycle run:\n"
            "  1. Re-train the champion to generate metrics.json:\n"
            "       python scripts/train.py --data data/synthetic_orders.json\n"
            "  2. Then train a challenger:\n"
            "       python scripts/train.py --target-dir models/challenger\n"
            "  3. Then re-run promote.py",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        chall_metrics = _load_metrics(challenger_dir, "Challenger")
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    champ_info = _load_model_info(champion_dir)
    chall_info = _load_model_info(challenger_dir)

    # ── Evaluate rules ───────────────────────────────────────────────────────
    rule_results = []  # (rule_id, passed, detail, explanation)
    all_passed = True
    for rule_id, desc, explanation in RULES:
        passed, detail = _evaluate_rule(rule_id, champ_metrics, chall_metrics)
        rule_results.append((rule_id, passed, detail, explanation))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {rule_id}: {detail}")
        if not passed:
            all_passed = False

    print()

    # ── Determine decision ───────────────────────────────────────────────────
    if dry_run:
        decision = "DRY RUN — WOULD BE PROMOTED" if all_passed else "DRY RUN — WOULD BE REJECTED"
    else:
        decision = "PROMOTED" if all_passed else "REJECTED"

    print(f"  => Decision: {decision}\n")

    # ── Build report ─────────────────────────────────────────────────────────
    report_md = _build_report(
        champ_metrics, chall_metrics,
        champ_info, chall_info,
        rule_results, decision, dry_run, timestamp,
    )

    decision_json = {
        "timestamp": timestamp,
        "decision": decision,
        "dry_run": dry_run,
        "champion": {
            "directory": str(champion_dir),
            "version": champ_info.get("model_version"),
            "trained_on": champ_info.get("trained_on"),
            "metrics": {k: v for k, v in champ_metrics.items() if k != "per_target"},
        },
        "challenger": {
            "directory": str(challenger_dir),
            "version": chall_info.get("model_version"),
            "trained_on": chall_info.get("trained_on"),
            "metrics": {k: v for k, v in chall_metrics.items() if k != "per_target"},
        },
        "rules": [
            {
                "id":          rule_id,
                "description": desc,
                "passed":      passed,
                "detail":      detail,
            }
            for rule_id, passed, detail, desc in [
                (r[0], r[1], r[2], RULES[i][1]) for i, r in enumerate(rule_results)
            ]
        ],
        "promoted": all_passed and not dry_run,
    }

    # ── Write artifacts (skip if dry-run) ────────────────────────────────────
    if not dry_run:
        report_dir.mkdir(parents=True, exist_ok=True)

        decision_path = report_dir / "promotion_decision.json"
        with open(decision_path, "w", encoding="utf-8") as fh:
            json.dump(decision_json, fh, indent=2)
        print(f"  [saved] {decision_path}")

        report_path = report_dir / "promotion_report.md"
        report_path.write_text(report_md, encoding="utf-8")
        print(f"  [saved] {report_path}")

        if all_passed:
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_slot = _archive_champion(champion_dir, archive_dir, champ_info)
            print(f"  [archived] previous champion -> {archive_slot}")
            _promote(champion_dir, challenger_dir)
            print(f"  [promoted] challenger -> {champion_dir}")
        else:
            print("  Champion unchanged. Challenger remains in models/challenger/.")
    else:
        # In dry-run, still print the report to stdout
        print("-" * 60)
        print(report_md)
        print("-" * 60)
        print("\n[DRY RUN] No files written. Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
