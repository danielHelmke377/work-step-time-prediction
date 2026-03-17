"""
train.py
========
Fully mixed pipeline — best-per-target classifier AND best-per-target regressor.

Stage 1 — Classifiers (NO oversampling, class weights only):
  Both LogReg and LightGBM are trained per target.
  Thresholds are F1-tuned on the validation set.
  The model with higher val F1 is used for test-set predictions.

Stage 2 — Regressors (same as model_best_per_target_reg.py):
  Winner from markdowns/regressor_selection.md per target.
  hailrepair: plain Ridge vs Ridge+winsorise, chosen by val MAE.
  Fallback = mean of positives if n_pos < 5.

Outputs:
  docs/markdowns/training_results.md
  models/two_stage_pipeline.pkl
"""

import argparse
import json
import pickle
import warnings
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split

from repair_order.config import OUTPUT_TARGETS
from repair_order.features import build_numeric_features, build_order_text, preprocess_positions

warnings.filterwarnings("ignore")

BAR = "=" * 70
def section(t): print(f"\n{BAR}\n  {t}\n{BAR}")


class RepairOrderTrainer:
    """Class-based pipeline for training the two-stage repair order predictor."""
    
    RANDOM_STATE = 42
    RIDGE_ALPHA = 10.0
    WINSORISE_PCT = 95
    
    BEST_REG = {
        "calibration":         "lgbm",
        "wheelmeasurement":    "lgbm",
        "bodymeasurement":     "lgbm",
        "dismounting":         "lgbm",
        "bodyrepair":          "ridge",
        "assembly":            "ridge",
        "plasticrepair":       "lgbm",
        "cleaning":            "lgbm",
        "paintingPreparation": "ridge",
        "paintingSpraying":    "lgbm",
        "paintingFinish":      "ridge",
        "hailrepair":          "ridge_auto",
        "glas":                "lgbm",
        "allTiresService":     "lgbm",
    }
    
    def __init__(self, data_path: Path, md_dir: Path, models_dir: Path):
        self.data_path = data_path
        self.md_dir = md_dir
        self.models_dir = models_dir
        
        self.md_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # State variables
        self.make_freq_lookup = {}
        self.tfidf_word = None
        self.tfidf_char = None
        self.numeric_features = []
        
        self.clf_best = {}
        self.clf_thr = {}
        self.clf_choice = {}
        
        self.reg_models = {}
        self.reg_choice = {}
        
        self.X_train = self.X_val = self.X_test = None
        self.Y_bin_train = self.Y_bin_val = self.Y_bin_test = None
        self.Y_reg_train = self.Y_reg_val = self.Y_reg_test = None
        
        self.report_df = None
        
    def load_and_preprocess(self):
        section("STEP 1 — Feature engineering")

        with open(self.data_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
            
        valid = [r for r in raw if "input" in r and "output" in r
                 and "calculatedPositions" in r["input"]]

        self.make_freq_lookup = pd.Series(
            [r["input"].get("make", "unknown").strip().upper() for r in valid]
        ).value_counts().to_dict()

        records_text, records_feats, records_target = [], [], []
        for r in valid:
            pos  = preprocess_positions(r["input"].get("calculatedPositions", []))
            make = r["input"].get("make", "unknown")
            records_text.append(build_order_text(pos))
            records_feats.append(build_numeric_features(pos, make, self.make_freq_lookup))
            records_target.append(
                {t: float(r["output"].get(t, 0) or 0) for t in OUTPUT_TARGETS})

        df_num     = pd.DataFrame(records_feats).fillna(0).astype(float)
        self.numeric_features = list(df_num.columns)
        
        df_targets = pd.DataFrame(records_target)
        Y_binary   = (df_targets > 0).astype(int)
        Y_reg      = df_targets.copy()

        n_active        = Y_binary.sum(axis=1).values
        complexity_band = pd.cut(n_active, bins=[-1, 3, 6, 9, 100],
                                 labels=["low", "medium", "high", "very_high"])
        indices = np.arange(len(df_num))
        idx_trainval, idx_test = train_test_split(
            indices, test_size=0.20, random_state=self.RANDOM_STATE, stratify=complexity_band)
        idx_train, idx_val = train_test_split(
            idx_trainval, test_size=0.25, random_state=self.RANDOM_STATE,
            stratify=complexity_band[idx_trainval])

        self.tfidf_word = TfidfVectorizer(ngram_range=(1, 2), max_features=2000, min_df=2,
                                          sublinear_tf=True, analyzer="word",
                                          token_pattern=r"(?u)\b[\w\+]+\b")
        self.tfidf_char = TfidfVectorizer(ngram_range=(3, 5), max_features=1000, min_df=3,
                                          sublinear_tf=True, analyzer="char_wb")
        X_word = self.tfidf_word.fit_transform(records_text)
        X_char = self.tfidf_char.fit_transform(records_text)
        X_all  = sparse.hstack([X_word, X_char, sparse.csr_matrix(df_num.values)], format="csr")

        self.X_train = X_all[idx_train]
        self.X_val   = X_all[idx_val]
        self.X_test  = X_all[idx_test]
        
        self.Y_bin_train = Y_binary.iloc[idx_train]
        self.Y_bin_val   = Y_binary.iloc[idx_val]
        self.Y_bin_test  = Y_binary.iloc[idx_test]
        self.Y_reg_train = Y_reg.iloc[idx_train]
        self.Y_reg_val   = Y_reg.iloc[idx_val]
        self.Y_reg_test  = Y_reg.iloc[idx_test]

        print(f"  Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)}")
        
    def _best_f1_thr(self, model, Xv, yv):
        probs = model.predict_proba(Xv)[:, 1]
        best_f1, best_t = -1.0, 0.5
        for thr in np.linspace(0.05, 0.95, 91):
            s = f1_score(yv, (probs >= thr).astype(int), zero_division=0)
            if s > best_f1:
                best_f1, best_t = s, thr
        return best_f1, best_t

    def train_classifiers(self):
        section("STEP 2 — Best-per-target classifiers (no oversampling)")

        self.clf_best = {}
        self.clf_thr = {}
        self.clf_choice = {}

        for t in OUTPUT_TARGETS:
            y_tr = self.Y_bin_train[t].values
            y_va = self.Y_bin_val[t].values
            n_pos = y_tr.sum()
            n_neg = len(y_tr) - n_pos
            w = max(1.0, n_neg / (n_pos + 1e-9))

            # Logistic Regression
            lr = LogisticRegression(C=1.0, max_iter=1000, class_weight={0: 1.0, 1: w},
                                    random_state=self.RANDOM_STATE)
            lr.fit(self.X_train, y_tr)

            # LightGBM classifier
            lc = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                     scale_pos_weight=w, random_state=self.RANDOM_STATE,
                                     verbose=-1, n_jobs=1)
            lc.fit(self.X_train, y_tr)

            f1_lr, thr_lr = self._best_f1_thr(lr, self.X_val, y_va)
            f1_lc, thr_lc = self._best_f1_thr(lc, self.X_val, y_va)

            if f1_lr >= f1_lc:
                self.clf_best[t], self.clf_thr[t], self.clf_choice[t] = lr, thr_lr, "logreg"
            else:
                self.clf_best[t], self.clf_thr[t], self.clf_choice[t] = lc, thr_lc, "lgbm"

            print(f"  {t:<22}  logreg F1={f1_lr:.4f}  lgbm F1={f1_lc:.4f}  → {self.clf_choice[t]}")

    def _make_lgbm_reg(self, n_pos):
        return lgb.LGBMRegressor(
            objective="regression", n_estimators=300, learning_rate=0.05,
            num_leaves=31, min_child_samples=max(5, n_pos // 10),
            random_state=self.RANDOM_STATE, verbose=-1, n_jobs=1,
        )

    def _make_ridge(self):
        return Ridge(alpha=self.RIDGE_ALPHA)

    def _val_mae_score(self, model, Xv_pos, yv_pos):
        if len(yv_pos) == 0:
            return float("inf")
        return mean_absolute_error(yv_pos, np.maximum(0, model.predict(Xv_pos)))

    def train_regressors(self):
        section("STEP 3 — Best-per-target regressors")

        self.reg_models = {}
        self.reg_choice = {}

        for t in OUTPUT_TARGETS:
            mask_tr = self.Y_bin_train[t].values == 1
            n_pos   = mask_tr.sum()
            rtype   = self.BEST_REG.get(t, "lgbm")

            if n_pos < 5:
                y_avail = self.Y_reg_train.loc[self.Y_bin_train[t] == 1, t]
                self.reg_models[t] = ("fallback", float(y_avail.mean()))
                self.reg_choice[t] = "fallback"
                print(f"  [fallback ] {t:<22}  n_pos={n_pos}")
                continue

            X_pos = self.X_train[mask_tr]
            y_pos = self.Y_reg_train.loc[self.Y_bin_train[t] == 1, t].values

            if rtype == "lgbm":
                m = self._make_lgbm_reg(n_pos)
                m.fit(X_pos, y_pos)
                self.reg_models[t] = m
                self.reg_choice[t] = "lgbm"
                print(f"  [lgbm     ] {t:<22}  n_pos={n_pos}")

            elif rtype == "ridge":
                m = self._make_ridge()
                m.fit(X_pos, y_pos)
                self.reg_models[t] = m
                self.reg_choice[t] = "ridge"
                print(f"  [ridge    ] {t:<22}  n_pos={n_pos}")

            elif rtype == "ridge_auto":
                cap    = np.percentile(y_pos, self.WINSORISE_PCT)
                y_wins = np.clip(y_pos, 0, cap)
                m_plain = self._make_ridge()
                m_wins  = self._make_ridge()
                m_plain.fit(X_pos, y_pos)
                m_wins.fit(X_pos, y_wins)

                mask_va   = self.Y_bin_val[t].values == 1
                X_va_pos  = self.X_val[mask_va]
                y_va_pos  = self.Y_reg_val.loc[self.Y_bin_val[t] == 1, t].values
                mae_plain = self._val_mae_score(m_plain, X_va_pos, y_va_pos)
                mae_wins  = self._val_mae_score(m_wins,  X_va_pos, y_va_pos)

                if mae_wins < mae_plain:
                    self.reg_models[t] = m_wins
                    self.reg_choice[t] = f"ridge+wins(cap={cap:.0f})"
                    print(f"  [ridge+win] {t:<22}  plain={mae_plain:.2f} wins={mae_wins:.2f} → winsorise")
                else:
                    self.reg_models[t] = m_plain
                    self.reg_choice[t] = "ridge"
                    print(f"  [ridge    ] {t:<22}  plain={mae_plain:.2f} wins={mae_wins:.2f} → plain")

    def evaluate_and_save_report(self):
        section("STEP 4 — Evaluate: fully mixed CLF + fully mixed REG")

        freq_weights = {t: float(self.Y_bin_test[t].mean()) for t in OUTPUT_TARGETS}
        total_fw     = sum(freq_weights.values())

        REF = {
            "baseline_fw_f1": 0.9387, "baseline_fw_mae": 2.78,
            "optimal_fw_f1":  0.9347, "optimal_fw_mae":  1.88,
            "bptr_fw_f1":     0.9347, "bptr_fw_mae":     0.96,
        }

        rows = []
        for t in OUTPUT_TARGETS:
            y_true_bin = self.Y_bin_test[t].values
            y_true_reg = self.Y_reg_test[t].values

            probs  = self.clf_best[t].predict_proba(self.X_test)[:, 1]
            y_bin  = (probs >= self.clf_thr[t]).astype(int)

            m = self.reg_models[t]
            y_dur     = np.full(self.X_test.shape[0], m[1]) if isinstance(m, tuple) else np.maximum(0.0, m.predict(self.X_test))
            y_reg_hat = np.where(y_bin == 1, y_dur, 0.0)

            rows.append({
                "target":   t,
                "clf":      self.clf_choice[t],
                "reg":      self.reg_choice[t],
                "freq_w":   round(freq_weights[t], 3),
                "n_pos":    int(y_true_bin.sum()),
                "Accuracy": round(accuracy_score(y_true_bin, y_bin), 4),
                "F1":       round(f1_score(y_true_bin, y_bin, zero_division=0), 4),
                "Recall":   round(recall_score(y_true_bin, y_bin, zero_division=0), 4),
                "Prec":     round(precision_score(y_true_bin, y_bin, zero_division=0), 4),
                "MAE":      round(mean_absolute_error(y_true_reg, y_reg_hat), 2),
            })

        df = pd.DataFrame(rows).set_index("target")
        self.report_df = df

        macro_f1  = round(df["F1"].mean(), 4)
        macro_rec = round(df["Recall"].mean(), 4)
        macro_acc = round(df["Accuracy"].mean(), 4)
        macro_mae = round(df["MAE"].mean(), 2)
        fw_f1  = round(sum(freq_weights[t] * df.loc[t, "F1"]      for t in OUTPUT_TARGETS) / total_fw, 4)
        fw_rec = round(sum(freq_weights[t] * df.loc[t, "Recall"]   for t in OUTPUT_TARGETS) / total_fw, 4)
        fw_acc = round(sum(freq_weights[t] * df.loc[t, "Accuracy"] for t in OUTPUT_TARGETS) / total_fw, 4)
        fw_mae = round(sum(freq_weights[t] * df.loc[t, "MAE"]      for t in OUTPUT_TARGETS) / total_fw, 2)

        print(f"\n  {'Target':<22} {'clf':>6} {'reg':>18} {'freq_w':>6} {'Acc':>7} {'F1':>7} {'Rec':>7} {'MAE':>7}")
        print("  " + "-" * 83)
        for t in OUTPUT_TARGETS:
            r = df.loc[t]
            print(f"  {t:<22} {r['clf']:>6} {r['reg']:>18} {r['freq_w']:>6.3f} "
                  f"{r['Accuracy']:>7.4f} {r['F1']:>7.4f} {r['Recall']:>7.4f} {r['MAE']:>7.2f}")

        print(f"\n  {'Metric':<28} {'Baseline':>10} {'TrueOpt':>10} {'BestReg':>10} {'FullMix':>10}")
        print(f"  {'-'*60}")
        print(f"  {'Macro F1':<28} {'0.8372':>10} {'0.8372':>10} {'0.8372':>10} {macro_f1:>10.4f}")
        print(f"  {'Macro Recall':<28} {'0.779':>10} {'0.8594':>10} {'0.8594':>10} {macro_rec:>10.4f}")
        print(f"  {'Macro Accuracy':<28} {'0.9380':>10} {'0.9380':>10} {'0.9380':>10} {macro_acc:>10.4f}")
        print(f"  {'Freq-weighted F1':<28} {REF['baseline_fw_f1']:>10.4f} {REF['optimal_fw_f1']:>10.4f} {REF['bptr_fw_f1']:>10.4f} {fw_f1:>10.4f}")
        print(f"  {'Freq-weighted Recall':<28} {'0.9460':>10} {'0.9412':>10} {'0.9412':>10} {fw_rec:>10.4f}")
        print(f"  {'Freq-weighted Accuracy':<28} {'0.9433':>10} {'0.9433':>10} {'0.9433':>10} {fw_acc:>10.4f}")
        print(f"  {'Macro MAE (hrs)':<28} {'3.51':>10} {'2.14':>10} {'0.70':>10} {macro_mae:>10.2f}")
        print(f"  {'Freq-weighted MAE (hrs)':<28} {REF['baseline_fw_mae']:>10.2f} {REF['optimal_fw_mae']:>10.2f} {REF['bptr_fw_mae']:>10.2f} {fw_mae:>10.2f}")

        # Write markdown report
        section("STEP 5 — Writing markdown report")

        tbl_rows = []
        for t in OUTPUT_TARGETS:
            r = df.loc[t]
            tbl_rows.append(
                f"| `{t}` | {r['clf']} | {r['reg']} | {r['n_pos']} | {r['freq_w']:.3f} "
                f"| {r['Accuracy']:.4f} | {r['F1']:.4f} | {r['Recall']:.4f} | {r['MAE']:.2f} |"
            )

        clf_summary = {}
        for t in OUTPUT_TARGETS:
            clf_summary[self.clf_choice[t]] = clf_summary.get(self.clf_choice[t], 0) + 1

        md = f"""# Fully Mixed Pipeline — Best-Per-Target CLF + Best-Per-Target REG

**Date:** {date.today()}

**Goal:** Use the best classifier and the best regressor independently per target,
both chosen by validation-set comparison. No oversampling, class weights only.

## Classifier Summary

| Model | n targets |
|---|---|
| Logistic Regression | {clf_summary.get('logreg', 0)} |
| LightGBM | {clf_summary.get('lgbm', 0)} |

## Comparison vs Previous Experiments

| Metric | Baseline | True Optimal | Best-per-REG | **Fully Mixed** |
|---|---|---|---|---|
| Macro F1 | 0.8372 | 0.8372 | 0.8372 | **{macro_f1}** |
| Macro Recall | 0.779 | 0.8594 | 0.8594 | **{macro_rec}** |
| Macro Accuracy | 0.9380 | 0.9380 | 0.9380 | **{macro_acc}** |
| Freq-weighted F1 | {REF['baseline_fw_f1']} | {REF['optimal_fw_f1']} | {REF['bptr_fw_f1']} | **{fw_f1}** |
| Freq-weighted Recall | 0.9460 | 0.9412 | 0.9412 | **{fw_rec}** |
| Freq-weighted Accuracy | 0.9433 | 0.9433 | 0.9433 | **{fw_acc}** |
| Macro MAE | 3.51 hrs | 2.14 hrs | 0.70 hrs | **{macro_mae} hrs** |
| **Freq-weighted MAE** | {REF['baseline_fw_mae']} hrs | {REF['optimal_fw_mae']} hrs | {REF['bptr_fw_mae']} hrs | **{fw_mae} hrs** |

## Per-target Results

| Target | clf | reg | n_pos | freq_w | Accuracy | F1 | Recall | MAE |
|--------|-----|-----|-------|--------|----------|----|--------|-----|
{chr(10).join(tbl_rows)}
| **MACRO** | | | | | **{macro_acc}** | **{macro_f1}** | **{macro_rec}** | **{macro_mae}** |
| **FREQ-WT** | | | | | **{fw_acc}** | **{fw_f1}** | **{fw_rec}** | **{fw_mae}** |
"""

        md_path = self.md_dir / "training_results.md"
        md_path.write_text(md, encoding="utf-8")
        print(f"\n  [saved] {md_path}")

    def save_pipeline(self):
        section("STEP 6 — Save pipeline")

        pipeline_out = {
            "tfidf_word": self.tfidf_word,
            "tfidf_char": self.tfidf_char,
            "numeric_features": self.numeric_features,
            "clf_models": {"logreg": {}, "lgbm": {}},
            "reg_models": {"ridge": {}, "lgbm": {}},
            "thresholds": {"logreg": {}, "lgbm": {}},
            "best_clf_per_target": self.clf_choice,
            "output_targets": OUTPUT_TARGETS,
            "make_freq_lookup": self.make_freq_lookup,
        }

        for t in OUTPUT_TARGETS:
            ctype = self.clf_choice[t]
            pipeline_out["clf_models"][ctype][t] = self.clf_best[t]
            pipeline_out["thresholds"][ctype][t] = self.clf_thr[t]
            
            rmodel = self.reg_models[t]
            rchoice = self.reg_choice[t]
            
            if rchoice == "fallback":
                pipeline_out["reg_models"]["lgbm"][t] = ("mean_fallback", rmodel[1])
            elif "ridge" in rchoice:
                pipeline_out["reg_models"]["ridge"][t] = rmodel
            else:
                pipeline_out["reg_models"]["lgbm"][t] = rmodel

        pkl_path = self.models_dir / "two_stage_pipeline.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump(pipeline_out, fh, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"  [saved] {pkl_path}")
        print("  Done.")
        
    def run(self):
        """Execute the full training pipeline."""
        self.load_and_preprocess()
        self.train_classifiers()
        self.train_regressors()
        self.evaluate_and_save_report()
        self.save_pipeline()


def main():
    parser = argparse.ArgumentParser(description="Train the fully mixed pipeline.")
    parser.add_argument("--data", type=str, default=str(Path(__file__).resolve().parent.parent / "data" / "orders_simplified_sample.json"),
                        help="Path to the simplified orders JSON file.")
    args, _ = parser.parse_known_args()
    
    root = Path(__file__).resolve().parent.parent
    data_path = Path(args.data)
    md_dir = root / "docs" / "markdowns"
    models_dir = root / "models"
    
    trainer = RepairOrderTrainer(data_path, md_dir, models_dir)
    trainer.run()


if __name__ == "__main__":
    main()
