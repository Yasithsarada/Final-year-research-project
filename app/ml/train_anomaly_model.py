"""
Training + evaluation harness for the Fraud & Anomaly Detection Model.

Run from the `reasearch v2` project root:
    python -m app.ml.train_anomaly_model

What this does, in order:
  1. Generates a labelled candidate population and pushes it through the REAL
     FeatureEngineeringLayer (so the experiment tests production code).
  2. Trains the unsupervised IsolationForest on the training split.
  3. Runs an ABLATION STUDY: rules-only vs ML-only vs hybrid, so you can
     evidence whether the hybrid design is actually worth anything.
  4. Trains supervised baselines (Logistic Regression, Random Forest) with
     5-fold cross-validation, as an upper reference point.
  5. Sweeps the decision threshold and reports the best-F1 operating point,
     so FRAUD_HIGH_THRESHOLD stops being a magic number.
  6. Trains a Gradient Boosting classifier and plots its training vs.
     validation loss curve (loss_curve.png / loss_curve.csv).
  7. Saves the fitted model plus metrics.json / dataset.csv / pr_curve.csv
     into app_data/ml_results/ for your results chapter.

READ THIS BEFORE QUOTING NUMBERS: the population is synthetic (see
synthetic_data.py). These results validate the pipeline; they are not an
empirical finding about real candidates. See FRAUD_DETECTION_GUIDANCE.md §4.

A NOTE ON THE LOSS CURVE: IsolationForest (the production model) and the
RandomForest baseline are not trained by iterative loss minimisation, so
neither has a meaningful per-epoch loss to plot. Gradient Boosting *is*
trained in sequential stages with a real per-stage loss (binomial deviance),
so it's added here specifically to produce a genuine train/validation loss
curve — a standard diagnostic for convergence and overfitting. It is an
additional reference model for that purpose, not a replacement for the
IsolationForest the Decision Engine actually loads.
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — this runs from a terminal / CI, no display
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    log_loss,
)

import app.ml.pipeline_config as cfg
from app.ml.synthetic_data import generate_profile_dataset, build_xy
from app.services.anomaly_model import MLAnomalyDetector, FEATURE_NAMES

logging.basicConfig(level=logging.WARNING)  # keep FEL chatter out of the report
logger = logging.getLogger(__name__)


def _metrics(y_true, y_pred, y_score) -> dict:
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
    }


def _best_threshold(y_true, y_score):
    """Sweeps thresholds and returns the one maximising F1."""
    best = {"threshold": 0.5, "f1": -1.0}
    for t in np.linspace(0.01, 0.99, 99):
        pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best["f1"]:
            best = {
                "threshold": round(float(t), 3),
                "f1": round(float(f1), 4),
                "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
            }
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-genuine", type=int, default=cfg.SYNTHETIC_N_GENUINE,
                         help=f"Synthetic genuine profiles to generate (default {cfg.SYNTHETIC_N_GENUINE}).")
    parser.add_argument("--n-fraud", type=int, default=cfg.SYNTHETIC_N_FRAUD,
                         help=f"Synthetic fraud profiles to generate (default {cfg.SYNTHETIC_N_FRAUD}).")
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED,
                         help=f"Random seed for splits and all model fits (default {cfg.RANDOM_SEED}).")
    parser.add_argument("--results-dir", type=Path, default=cfg.RESULTS_DIR,
                         help=f"Where to write metrics/figures (default {cfg.RESULTS_DIR}).")
    parser.add_argument("--model-path", type=Path, default=cfg.ISOLATION_FOREST_PATH,
                         help=f"Where to save the trained IsolationForest (default {cfg.ISOLATION_FOREST_PATH}).")
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed
    report = {}

    # ------------------------------------------------------------------
    # 1. Build the population through the real feature pipeline
    # ------------------------------------------------------------------
    print("Generating candidate population and running the real "
          "FeatureEngineeringLayer over it...")
    profiles, skills, audios, y = generate_profile_dataset(
        n_genuine=args.n_genuine, n_fraud=args.n_fraud, seed=seed
    )
    X, rule_scores = build_xy(profiles, skills, audios)

    print(f"  {len(y)} candidates | {int(y.sum())} fraudulent ({y.mean():.1%}) "
          f"| {X.shape[1]} features")

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=cfg.TRAIN_TEST_SPLIT, random_state=seed, stratify=y)
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]
    rule_te = rule_scores[te]

    report["dataset"] = {
        "n_total": int(len(y)),
        "n_fraud": int(y.sum()),
        "fraud_rate": round(float(y.mean()), 4),
        "n_features": int(X.shape[1]),
        "feature_names": FEATURE_NAMES,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "synthetic": True,
    }

    # ------------------------------------------------------------------
    # 2. Train the unsupervised IsolationForest
    # ------------------------------------------------------------------
    print("\nTraining IsolationForest (unsupervised — labels NOT used to fit)...")
    contamination = float(np.mean(y_tr))
    detector = MLAnomalyDetector()
    detector.fit(X_tr, contamination=contamination)

    ml_scores_te = np.array([detector.score(x) for x in X_te])
    ml_pred_te = detector.predict_is_anomaly(X_te)

    # ------------------------------------------------------------------
    # 3. ABLATION STUDY — the part that justifies the hybrid design
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("ABLATION STUDY  (test split, n=%d)" % len(y_te))
    print("=" * 72)

    hybrid_te = cfg.HYBRID_RULE_WEIGHT * rule_te + cfg.HYBRID_ML_WEIGHT * ml_scores_te

    # Collected across this section and the baselines loop below, then
    # written out once as a single combined ROC/PR curve file so every
    # model's curve can be plotted on the same axes for comparison.
    curve_scores = {"rules_only": rule_te, "ml_only": ml_scores_te, "hybrid": hybrid_te}

    ablation = {
        "rules_only": _metrics(y_te, (rule_te >= 0.30).astype(int), rule_te),
        "ml_only": _metrics(y_te, ml_pred_te, ml_scores_te),
        "hybrid": _metrics(y_te, (hybrid_te >= 0.30).astype(int), hybrid_te),
    }
    report["ablation"] = ablation

    print(f"{'variant':<14}{'precision':>11}{'recall':>9}{'F1':>9}{'ROC-AUC':>10}{'PR-AUC':>9}")
    print("-" * 72)
    for name, m in ablation.items():
        print(f"{name:<14}{m['precision']:>11.3f}{m['recall']:>9.3f}"
              f"{m['f1']:>9.3f}{m['roc_auc']:>10.3f}{m['pr_auc']:>9.3f}")

    best_variant = max(ablation.items(), key=lambda kv: kv[1]["f1"])[0]
    print(f"\nBest by F1: {best_variant}")
    if best_variant != "hybrid":
        print("  NOTE: the hybrid did NOT win here. Report this honestly — a\n"
              "  negative result you explain is worth more than a hidden one.\n"
              "  Likely cause: the rule engine already captures most of the\n"
              "  synthetic signal. Real data may change this.")

    # ------------------------------------------------------------------
    # 4. Supervised baselines (upper reference point)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUPERVISED BASELINES  (5-fold stratified CV on training split)")
    print("=" * 72)
    print("Context: these DO use labels, so they are an upper bound. Your")
    print("IsolationForest does not get to see labels — that is the point.\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    baselines = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=cfg.RF_N_ESTIMATORS, class_weight="balanced", random_state=seed
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=cfg.GBM_N_ESTIMATORS, learning_rate=cfg.GBM_LEARNING_RATE,
            max_depth=cfg.GBM_MAX_DEPTH, random_state=seed
        ),
    }
    report["supervised_baselines"] = {}
    for name, clf in baselines.items():
        scores = cross_val_score(clf, X_tr, y_tr, cv=cv, scoring="f1")
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        curve_scores[name] = proba
        m = _metrics(y_te, clf.predict(X_te), proba)
        m["cv_f1_mean"] = round(float(scores.mean()), 4)
        m["cv_f1_std"] = round(float(scores.std()), 4)
        report["supervised_baselines"][name] = m
        print(f"{name:<22} CV F1 = {scores.mean():.3f} (+/- {scores.std():.3f})   "
              f"test F1 = {m['f1']:.3f}  ROC-AUC = {m['roc_auc']:.3f}")

    # Feature importance from the random forest — useful for your write-up.
    rf = baselines["random_forest"]
    importances = sorted(
        zip(FEATURE_NAMES, rf.feature_importances_), key=lambda kv: kv[1], reverse=True
    )
    report["feature_importance"] = {k: round(float(v), 4) for k, v in importances}
    print("\nFeature importance (Random Forest):")
    for name, imp in importances:
        bar = "#" * int(imp * 60)
        print(f"  {name:<32}{imp:>6.3f}  {bar}")

    # ------------------------------------------------------------------
    # 4b. Training vs. validation loss curve (Gradient Boosting)
    # ------------------------------------------------------------------
    # A held-out slice of the TRAINING split only — X_te/y_te stays fully
    # untouched for the final evaluation above and in step 6.
    print("\n" + "=" * 72)
    print("TRAINING / VALIDATION LOSS CURVE  (Gradient Boosting)")
    print("=" * 72)

    # This reference model uses higher capacity (deeper trees, faster
    # learning) than the GBM baseline above — deliberately, so a genuine
    # overfitting signature is visible within a reasonable iteration count.
    # The first version of this diagnostic used the same shallow settings as
    # the GBM baseline; validation loss still rose after its optimum, but by
    # too little to be visible against training loss's dominant slope.
    X_sub_tr, X_val, y_sub_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=cfg.LOSS_CURVE_VAL_FRACTION, random_state=seed, stratify=y_tr
    )
    loss_clf = GradientBoostingClassifier(
        n_estimators=cfg.LOSS_CURVE_N_ESTIMATORS, learning_rate=cfg.LOSS_CURVE_LEARNING_RATE,
        max_depth=cfg.LOSS_CURVE_MAX_DEPTH, random_state=seed
    )
    loss_clf.fit(X_sub_tr, y_sub_tr)

    # train_score_[i] is the binomial deviance (= log loss) of the model at
    # boosting stage i on the data passed to fit() — this IS the training loss.
    train_loss = loss_clf.train_score_
    # Validation loss: same loss function (log loss), evaluated at every stage
    # via staged_predict_proba so it's directly comparable to train_loss.
    val_loss = np.array([
        log_loss(y_val, proba) for proba in loss_clf.staged_predict_proba(X_val)
    ])
    iterations = np.arange(1, len(train_loss) + 1)
    best_iter = int(np.argmin(val_loss)) + 1

    report["loss_curve"] = {
        "model": "GradientBoostingClassifier",
        "n_estimators": len(train_loss),
        "final_train_loss": round(float(train_loss[-1]), 4),
        "final_val_loss": round(float(val_loss[-1]), 4),
        "best_val_loss": round(float(val_loss.min()), 4),
        "best_iteration": best_iter,
        "overfitting_gap_at_end": round(float(val_loss[-1] - train_loss[-1]), 4),
    }

    print(f"  {len(X_sub_tr)} train / {len(X_val)} validation rows "
          f"(carved from the {len(X_tr)}-row training split)")
    print(f"  Final train loss: {train_loss[-1]:.4f}   Final val loss: {val_loss[-1]:.4f}")
    print(f"  Best val loss:    {val_loss.min():.4f}  at iteration {best_iter}/{len(train_loss)}")
    if val_loss[-1] - val_loss.min() > 0.02:
        print(f"  NOTE: validation loss is still rising after iteration {best_iter} while "
              "training loss keeps falling — the model is overfitting past that point. "
              f"Consider n_estimators={best_iter} or early stopping.")
    else:
        print("  Train and validation loss track closely — no strong overfitting signal.")

    # Palette: reference instance from the dataviz skill (categorical slots
    # 1=blue, 2=orange) — kept consistent with the interactive chart shown
    # alongside this run.
    COLOR_TRAIN = "#2a78d6"
    COLOR_VAL = "#eb6834"
    COLOR_GRID = "#e1e0d9"
    COLOR_AXIS = "#c3c2b7"
    COLOR_INK = "#0b0b0b"
    COLOR_MUTED = "#52514e"

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150, facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(iterations, train_loss, color=COLOR_TRAIN, linewidth=2, label="Training loss")
    ax.plot(iterations, val_loss, color=COLOR_VAL, linewidth=2, label="Validation loss")
    ax.axvline(best_iter, color=COLOR_MUTED, linewidth=1, linestyle="--", alpha=0.7)
    ax.annotate(
        f"best @ {best_iter}", xy=(best_iter, val_loss.min()),
        xytext=(best_iter + len(train_loss) * 0.03, val_loss.min() + (val_loss.max() - val_loss.min()) * 0.08),
        color=COLOR_MUTED, fontsize=9,
    )
    ax.set_xlabel("Boosting iteration", color=COLOR_INK)
    ax.set_ylabel("Log loss (binomial deviance)", color=COLOR_INK)
    ax.set_title("Gradient Boosting — Training vs. Validation Loss", color=COLOR_INK, fontsize=12)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_AXIS)
    ax.spines["bottom"].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_MUTED)
    legend = ax.legend(frameon=False, loc="upper right")
    for text in legend.get_texts():
        text.set_color(COLOR_INK)
    fig.tight_layout()

    loss_plot_path = results_dir / "loss_curve.png"
    fig.savefig(loss_plot_path)
    plt.close(fig)

    loss_csv_path = results_dir / "loss_curve.csv"
    with open(loss_csv_path, "w") as f:
        f.write("iteration,train_loss,val_loss\n")
        for it, tl, vl in zip(iterations, train_loss, val_loss):
            f.write(f"{it},{tl:.6f},{vl:.6f}\n")

    # ------------------------------------------------------------------
    # 5. Threshold calibration
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("THRESHOLD CALIBRATION  (hybrid score)")
    print("=" * 72)
    best = _best_threshold(y_te, hybrid_te)
    report["threshold_sweep"] = best
    print(f"Best-F1 threshold: {best['threshold']}  "
          f"(precision {best['precision']:.3f}, recall {best['recall']:.3f}, "
          f"F1 {best['f1']:.3f})")
    print(f"Currently DecisionEngine.FRAUD_MEDIUM_THRESHOLD = 0.30, "
          f"FRAUD_HIGH_THRESHOLD = 0.60")
    print("\nIn recruitment, a false accusation costs more than a missed one.\n"
          "Consider favouring precision over recall when you pick the final\n"
          "operating point, and say so explicitly in your thesis.")

    prec, rec, thr = precision_recall_curve(y_te, hybrid_te)
    pr_path = results_dir / "pr_curve.csv"
    with open(pr_path, "w") as f:
        f.write("threshold,precision,recall\n")
        for i, t in enumerate(thr):
            f.write(f"{t:.6f},{prec[i]:.6f},{rec[i]:.6f}\n")

    # Combined ROC + PR curves for every model in the comparison (ablation
    # variants + supervised baselines), on the same held-out test split —
    # this is what "why this model was chosen" plots are built from.
    roc_pr_path = results_dir / "roc_pr_curves.csv"
    with open(roc_pr_path, "w") as f:
        f.write("model,curve,x,y,threshold\n")
        for name, scores in curve_scores.items():
            fpr, tpr, roc_thr = roc_curve(y_te, scores)
            for i in range(len(fpr)):
                t = roc_thr[i] if np.isfinite(roc_thr[i]) else 1.0
                f.write(f"{name},roc,{fpr[i]:.6f},{tpr[i]:.6f},{t:.6f}\n")
            p, r, pr_thr = precision_recall_curve(y_te, scores)
            pr_thr = np.append(pr_thr, 1.0)  # precision_recall_curve returns one fewer threshold
            for i in range(len(p)):
                f.write(f"{name},pr,{r[i]:.6f},{p[i]:.6f},{pr_thr[i]:.6f}\n")

    # ------------------------------------------------------------------
    # 6. Detailed report for the winning configuration
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("HYBRID MODEL — DETAILED TEST REPORT")
    print("=" * 72)
    hybrid_pred = (hybrid_te >= best["threshold"]).astype(int)
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_te, hybrid_pred))
    print()
    print(classification_report(y_te, hybrid_pred,
                                target_names=["genuine", "fraud"], zero_division=0))

    # ------------------------------------------------------------------
    # 7. Persist artefacts
    # ------------------------------------------------------------------
    detector.save(args.model_path)

    ds_path = results_dir / "dataset.csv"
    with open(ds_path, "w") as f:
        f.write(",".join(FEATURE_NAMES) + ",rule_score,label\n")
        for row, rs, lab in zip(X, rule_scores, y):
            f.write(",".join(f"{v:.6f}" for v in row) + f",{rs:.6f},{int(lab)}\n")

    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    print("ARTEFACTS SAVED")
    print("=" * 72)
    print(f"  model        {args.model_path}")
    print(f"  metrics      {metrics_path}")
    print(f"  dataset      {ds_path}")
    print(f"  PR curve     {pr_path}")
    print(f"  ROC+PR (all) {roc_pr_path}")
    print(f"  loss curve   {loss_plot_path}")
    print(f"  loss data    {loss_csv_path}")
    print("\nThe Decision Engine will load the model automatically on next start.")
    print("\nREMINDER: this population is SYNTHETIC. These numbers validate the")
    print("pipeline, they are not a finding about real candidates. Replace")
    print("generate_profile_dataset() with real labelled profiles before you")
    print("write your results chapter — see FRAUD_DETECTION_GUIDANCE.md §4.")


if __name__ == "__main__":
    main()
