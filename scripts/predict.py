"""
predict.py — Inference Script
================================
Loads the saved two-stage pipeline and predicts repair work steps
from a raw JSON order (or an example pulled from the dataset).

Usage:
    # Predict from a JSON file containing a single order:
    python scripts/predict.py --input my_order.json

    # Predict from a specific record in the training data (by index):
    python scripts/predict.py --demo 42

    # Run a batch of N random demo orders and print a summary table:
    python scripts/predict.py --batch 10

    # Evaluate against true labels for a demo record:
    python scripts/predict.py --demo 42 --evaluate
"""

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

from repair_order.config import KEYWORD_FLAGS, TARGET_LABELS, TARGET_KEYWORD_MAP
from repair_order.pipeline import load_pipeline as lib_load_pipeline, predict_order

warnings.filterwarnings("ignore", message="X does not have valid feature names")

BAR = "=" * 68

class RepairOrderPredictor:
    """Class-based predictor for processing repair orders through the ML pipeline."""

    def __init__(self, pipeline_path: Path, data_path: Path = None):
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
            
        print(f"  Loading pipeline from: {pipeline_path}")
        self.pipeline = lib_load_pipeline(pipeline_path)
        print(f"  Pipeline loaded. Targets: {len(self.pipeline['output_targets'])}")
        
        self.compiled_keywords = {k: re.compile(v, re.IGNORECASE) for k, v in KEYWORD_FLAGS.items()}
        
        self.dataset = None
        if data_path and data_path.exists():
            with open(data_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.dataset = [r for r in raw if "input" in r and "output" in r]

    def _explain_prediction(self, result, record):
        """
        For each active target, return the top matching position texts
        and which keyword flags triggered.
        """
        positions    = record["input"]["calculatedPositions"]
        explanations = {}

        for t in result["active_targets"]:
            kw_names   = TARGET_KEYWORD_MAP.get(t, [])
            triggered  = []
            for kw in kw_names:
                if result["_feats"].get(kw, 0):
                    triggered.append(kw.replace("kw_", ""))

            matching_texts = []
            for p in positions:
                txt = (p.get("text") or "").strip()
                if not txt:
                    continue
                for kw in kw_names:
                    pat = self.compiled_keywords.get(kw)
                    if pat and pat.search(txt):
                        matching_texts.append(txt[:80])
                        break
            matching_texts = list(dict.fromkeys(matching_texts))[:3]

            explanations[t] = {
                "triggered_keywords": triggered,
                "matching_positions": matching_texts,
            }

        return explanations

    def _print_report(self, result, explanations=None, true_output=None):
        print(f"\n{BAR}")
        print("  WORK STEP TIME PREDICTION REPORT")
        print(BAR)
        print(f"  Make            : {result['make']}")
        print(f"  Line items      : {result['n_positions']}")
        print(f"  Total input time: {result['total_input_time_hrs']:.1f} hrs")
        print(f"  Total input cost: EUR {result['total_input_price_eur']:,.2f}")
        print(f"  Inference time  : {result['elapsed_ms']:.1f} ms")

        print(f"\n  {'TARGET':<30} {'ACTIVE':>7} {'PROB':>7} {'PRED(hrs)':>10}"
              + ("  TRUE(hrs)" if true_output else ""))
        print("  " + "-" * (60 + (10 if true_output else 0)))

        targets = result["predictions"]
        for t in result["_order"]:
            p      = targets[t]
            label  = TARGET_LABELS.get(t, t)
            active = "YES" if p["active"] else "---"
            prob   = f"{p['prob']:.2f}"
            hrs   = f"{p['predicted_hours']:.2f}" if p["active"] else "   0.00"
            line   = f"  {label:<30} {active:>7} {prob:>7} {hrs:>10}"
            if true_output is not None:
                true_val = true_output.get(t, 0) or 0
                true_str = f"{float(true_val):>9.2f}"
                correct  = "" if (p["active"] == (float(true_val) > 0)) else "  <-- MISS"
                line    += f"  {true_str}{correct}"
            print(line)

        print(f"\n  Total predicted repair time: {result['total_predicted_hours']:.2f} hrs")

        if true_output is not None:
            true_total = sum(float(v or 0) for v in true_output.values())
            error      = abs(result["total_predicted_hours"] - true_total)
            print(f"  True total repair time     : {true_total:.2f} hrs")
            print(f"  Total time error           : {error:.2f} hrs")

        if explanations:
            print("\n  EXPLANATION - Why each work step was predicted:")
            print("  " + "-" * 60)
            for t in result["active_targets"]:
                label = TARGET_LABELS.get(t, t)
                exp   = explanations.get(t, {})
                kwds  = ", ".join(exp.get("triggered_keywords", [])) or "text pattern"
                print(f"\n  [{label}]")
                print(f"    Keywords matched : {kwds}")
                for pos in exp.get("matching_positions", []):
                    print(f"    Position text    : \"{pos}\"")

        print(f"\n{BAR}\n")

    def _print_batch_summary(self, results, true_outputs=None):
        print(f"\n{BAR}")
        print(f"  BATCH PREDICTION SUMMARY  ({len(results)} orders)")
        print(BAR)
        print(f"  {'#':<4} {'Make':<20} {'Items':>5} "
              f"{'Pred Steps':>10} {'Pred Total(hrs)':>15}"
              + (f"  {'True Total(hrs)':>15} {'Error(hrs)':>11}" if true_outputs else ""))
        print("  " + "-" * (58 + (28 if true_outputs else 0)))

        errors = []
        for i, res in enumerate(results):
            pred_total = res["total_predicted_hours"]
            line = (f"  {i:<4} {res['make']:<20} {res['n_positions']:>5} "
                    f"{len(res['active_targets']):>10} {pred_total:>15.2f}")
            if true_outputs:
                true_total = sum(float(v or 0) for v in true_outputs[i].values())
                err        = abs(pred_total - true_total)
                errors.append(err)
                line += f"  {true_total:>15.2f} {err:>11.2f}"
            print(line)

        if errors:
            print(f"\n  Mean absolute error (total time): {np.mean(errors):.2f} hrs")
            print(f"  Median absolute error           : {np.median(errors):.2f} hrs")
            print(f"  Max absolute error              : {np.max(errors):.2f} hrs")
        print(f"\n{BAR}\n")

    def predict_record(self, record, evaluate=False, explain=True):
        """Predicts a single raw JSON record dict."""
        result = predict_order(record, self.pipeline)
        result["_order"] = self.pipeline["output_targets"]
        explanations = self._explain_prediction(result, record) if explain else None
        true_out = record.get("output", None) if evaluate else None
        self._print_report(result, explanations=explanations, true_output=true_out)

    def predict_file(self, json_path: Path, explain=True):
        """Predicts from a single JSON file."""
        if not json_path.exists():
            print(f"[ERROR] File not found: {json_path}")
            sys.exit(1)
        with open(json_path, "r", encoding="utf-8") as f:
            record = json.load(f)
        if isinstance(record, list):
            record = record[0]
        self.predict_record(record, evaluate=False, explain=explain)

    def predict_demo(self, index: int, evaluate=False, explain=True):
        """Pulls a single record from the loaded dataset and predicts."""
        if not self.dataset:
            print("[ERROR] No dataset loaded for demo.")
            sys.exit(1)
        idx = index % len(self.dataset)
        record = self.dataset[idx]
        print(f"  Demo record index: {idx} / {len(self.dataset)-1}")
        self.predict_record(record, evaluate=evaluate, explain=explain)

    def predict_batch(self, batch_size: int, evaluate=False, explain=True):
        """Pulls a batch of random records from the loaded dataset."""
        if not self.dataset:
            print("[ERROR] No dataset loaded for batch prediction.")
            sys.exit(1)
            
        rng = np.random.default_rng(42)
        indices = rng.choice(len(self.dataset), size=min(batch_size, len(self.dataset)), replace=False)
        results = []
        true_outputs = [] if evaluate else None
        
        for idx in indices:
            record = self.dataset[idx]
            res = predict_order(record, self.pipeline)
            res["_order"] = self.pipeline["output_targets"]
            results.append(res)
            if evaluate:
                true_outputs.append(record["output"])
                
        self._print_batch_summary(results, true_outputs=true_outputs)

        print(f"  Detailed report for first batch record (index {indices[0]}):")
        exp = self._explain_prediction(results[0], self.dataset[indices[0]]) if explain else None
        true_out = self.dataset[indices[0]]["output"] if evaluate else None
        self._print_report(results[0], explanations=exp, true_output=true_out)


def main():
    root = Path(__file__).resolve().parent.parent
    
    parser = argparse.ArgumentParser(description="Predict repair work steps from a JSON order.")
    parser.add_argument("--data", type=str, default=str(root / "data" / "orders_simplified_sample.json"))
    parser.add_argument("--model", type=str, default=str(root / "models" / "two_stage_pipeline.pkl"))
    parser.add_argument("--input", type=str, help="Path to a JSON file containing a single order dict.")
    parser.add_argument("--demo", type=int, help="Index of a record in the training data to use as demo.")
    parser.add_argument("--batch", type=int, help="Run N random records from the dataset as a batch demo.")
    parser.add_argument("--evaluate", action="store_true", help="Show true labels alongside predictions (demo/batch only).")
    parser.add_argument("--explain", action="store_true", default=True, help="Show keyword-based explanation (default: True).")
    parser.add_argument("--no-explain", dest="explain", action="store_false")
    args = parser.parse_args()

    print(f"\n{BAR}")
    print("  Work Step Time Inference Engine")
    print(BAR)

    data_path = Path(args.data)
    model_path = Path(args.model)
    
    predictor = RepairOrderPredictor(pipeline_path=model_path, data_path=data_path)

    if args.input:
        predictor.predict_file(Path(args.input), explain=args.explain)
    elif args.demo is not None:
        predictor.predict_demo(args.demo, evaluate=args.evaluate, explain=args.explain)
    elif args.batch is not None:
        predictor.predict_batch(args.batch, evaluate=args.evaluate, explain=args.explain)
    else:
        print("  [WARN] No action specified. Use --input, --demo, or --batch.")
        parser.print_help()


if __name__ == "__main__":
    main()
