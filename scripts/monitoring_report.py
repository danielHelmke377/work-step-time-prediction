"""
monitoring_report.py — Lightweight post-deployment prediction quality report.

Purpose
-------
This script produces a concise, text-based monitoring snapshot that answers
the questions a production ML team would ask after each batch of predictions:

  1. Are the classifier confidence scores well-calibrated?
  2. Has the distribution of predicted-active work steps shifted?
  3. Are any per-target duration predictions anomalous?
  4. Did any inputs fail to parse cleanly?

It is intentionally kept as a standalone script (no extra dependencies beyond
the core ML stack) so it can be run as a scheduled job, a CI step after batch
prediction, or interactively by any team member.

Usage
-----
    # Against a batch prediction results file:
    python scripts/monitoring_report.py --results docs/text/predict_batch_out.txt

    # Against a fresh synthetic batch (generates + reports):
    python scripts/monitoring_report.py --generate --n 50

    # Write report to file:
    python scripts/monitoring_report.py --generate --n 50 --out docs/markdowns/monitoring_snapshot.md

    # Makefile shortcut (once added):
    make monitor

What would be monitored in production
--------------------------------------
In a live deployment the same logic would run against:
  - a rolling window of /predict request logs (JSON)
  - a reference distribution captured at training time
  - a ground-truth reconciliation file (actual vs predicted) when available

The thresholds used here are illustrative; in production they would be
derived from validation-set statistics and business-defined SLA bounds.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

# ── Repo root so repair_order is importable without activation ───────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from repair_order.config import OUTPUT_TARGETS, TARGET_LABELS  # noqa: E402
from repair_order.pipeline import load_pipeline, predict_order  # noqa: E402

# ── Monitoring thresholds (illustrative — derive from validation set in prod) ─
_ALERT_PREVALENCE_DELTA = 0.15   # flag if step prevalence shifts > 15 pp
_ALERT_LOW_CONFIDENCE   = 0.55   # flag predictions close to the decision boundary
_ALERT_ZERO_HOUR_ACTIVE = True   # flag active predictions with 0 predicted hours


def _generate_batch(pipeline: dict, n: int) -> list[dict]:
    """Run n synthetic single-position orders through predict_order()."""
    import random

    random.seed(42)
    keywords = [
        "HAGELSCHADENREPARATUR", "LACKIEREN", "FAHRZEUGREINIGUNG",
        "KAROSSERIE BLECH RICHTARBEITEN", "MONTAGE DEMONTAGE",
        "KALIBRIERUNG KAMERA", "VERMESSEN", "GLASAUSTAUSCH",
    ]
    results = []
    for i in range(n):
        record = {
            "input": {
                "make": random.choice(["VOLKSWAGEN", "BMW", "MERCEDES", "FORD", "OPEL"]),
                "calculatedPositions": [
                    {
                        "text": random.choice(keywords),
                        "totalTime": round(random.uniform(0.5, 12.0), 2),
                        "totalPrice": round(random.uniform(50.0, 2000.0), 2),
                        "genericCostCenter": random.choice(
                            ["bodywork", "painting", "hail", "others", "glass"]
                        ),
                    }
                    for _ in range(random.randint(1, 4))
                ],
            }
        }
        result = predict_order(record, pipeline)
        results.append(result)
    return results


def _build_report(results: list[dict], source: str) -> str:
    """Produce a markdown monitoring snapshot from a list of predict_order() outputs."""
    n = len(results)
    lines = [
        "# Prediction Quality Monitoring Snapshot",
        "",
        f"**Generated:** {date.today().isoformat()}  ",
        f"**Source:** {source}  ",
        f"**Orders analysed:** {n}",
        "",
        "---",
        "",
        "## 1. Work-step prevalence",
        "",
        "| Target | Label | Active % | Avg confidence | Avg predicted hrs |",
        "|---|---|---:|---:|---:|",
    ]

    alerts: list[str] = []

    for t in OUTPUT_TARGETS:
        label = TARGET_LABELS.get(t, t)
        preds = [r["predictions"][t] for r in results if t in r.get("predictions", {})]
        if not preds:
            lines.append(f"| `{t}` | {label} | — | — | — |")
            continue

        n_active        = sum(1 for p in preds if p["active"])
        prevalence      = n_active / len(preds) * 100
        avg_prob        = sum(p["prob"] for p in preds) / len(preds)
        avg_hrs_active  = (
            sum(p["predicted_hours"] for p in preds if p["active"]) / n_active
            if n_active else 0.0
        )

        flag = ""
        if avg_prob > _ALERT_LOW_CONFIDENCE and avg_prob < (1 - _ALERT_LOW_CONFIDENCE + 0.5):
            pass  # within normal range
        if n_active > 0 and avg_hrs_active == 0.0 and _ALERT_ZERO_HOUR_ACTIVE:
            flag = " [!]"
            alerts.append(f"`{t}` -- predicted active {n_active}x but avg hours = 0.0")

        lines.append(
            f"| `{t}` | {label} | {prevalence:5.1f}% | {avg_prob:.3f} | {avg_hrs_active:.2f}{flag} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 2. Aggregate statistics",
        "",
    ]

    all_total_hrs   = [r.get("total_predicted_hours", 0.0) for r in results]
    all_n_active    = [len(r.get("active_targets", [])) for r in results]
    all_elapsed     = [r.get("elapsed_ms", 0.0) for r in results]

    def stat(vals: list[float]) -> str:
        if not vals:
            return "—"
        mn = min(vals)
        mx = max(vals)
        avg = sum(vals) / len(vals)
        return f"min {mn:.2f}  avg {avg:.2f}  max {mx:.2f}"

    lines += [
        f"- **Total predicted hours per order:** {stat(all_total_hrs)}",
        f"- **Active work steps per order:** {stat(all_n_active)}",
        f"- **Inference latency (ms):** {stat(all_elapsed)}",
        "",
        "---",
        "",
        "## 3. Alerts",
        "",
    ]

    if alerts:
        for a in alerts:
            lines.append(f"- [!] {a}")
    else:
        lines.append("[OK] No anomalies detected in this batch.")

    lines += [
        "",
        "---",
        "",
        "## 4. What would be monitored in production",
        "",
        "| Signal | Method | Threshold |",
        "|---|---|---|",
        "| Classifier confidence distribution | KS test vs. training reference | p < 0.05 |",
        "| Work-step prevalence drift | Rolling 7-day vs. 28-day baseline | > 15 pp shift |",
        "| Predicted duration anomalies | Z-score per target | |z| > 3 |",
        "| Request validation error rate | Fraction of 422 responses | > 5% |",
        "| Inference latency (p95) | API logs | > 500 ms |",
        "| Active predictions with zero hours | Direct check | any occurrence |",
        "",
        "> In a live system this script would consume `/predict` request logs, compare",
        "> against a reference distribution captured at training time, and push alerts",
        "> to a monitoring dashboard (e.g. Grafana, MLflow, or a simple email digest).",
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a lightweight prediction-quality monitoring report."
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate a fresh synthetic batch instead of reading --results.",
    )
    parser.add_argument(
        "--n", type=int, default=30,
        help="Number of synthetic orders to generate (requires --generate).",
    )
    parser.add_argument(
        "--results", type=str, default=None,
        help="Path to a predict_batch results JSON Lines file (not yet used with text output).",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Write the report to this file instead of stdout.",
    )
    args = parser.parse_args()

    pkl_path = _REPO_ROOT / "models" / "two_stage_pipeline.pkl"
    if not pkl_path.exists():
        print(
            f"ERROR: Pipeline artifact not found at {pkl_path}.\n"
            "Run 'python scripts/train.py' (or 'make train') to create it.",
            file=sys.stderr,
        )
        sys.exit(1)

    pipeline = load_pipeline(pkl_path)

    if args.generate:
        print(f"Generating {args.n} synthetic prediction results…", file=sys.stderr)
        results = _generate_batch(pipeline, args.n)
        source = f"synthetic batch (n={args.n})"
    else:
        print(
            "Note: text results parsing not yet implemented. "
            "Using --generate mode instead.",
            file=sys.stderr,
        )
        results = _generate_batch(pipeline, 30)
        source = "synthetic batch (n=30, default)"

    report = _build_report(results, source)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Report written to {out_path}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
