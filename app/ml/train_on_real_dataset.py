"""
Training + evaluation harness for the Fraud & Anomaly Detection Model —
REAL LABELLED DATA (datasets/interview_cv_github_fraud_dataset.csv).

Run from the `reasearch v2` project root:
    python -m app.ml.train_on_real_dataset

Unlike train_anomaly_model.py (synthetic population, no real ground truth),
this dataset is real: 8 real, verifiable GitHub accounts, each interviewed
across 3 cases (strong_alignment / partial_alignment / strong_misalignment),
24 rows total, perfectly balanced 12 genuine / 12 fraud (Is_Fraud). GitHub
evidence is pre-fetched and cached per row (GitHub_Evidence column) — no
live API calls needed.

WHAT THIS DOES:
  1. Parses the cached GitHub_Evidence JSON per row (real MSR data).
  2. Calls Gemini once per row to compare the interview Transcript + Resume
     against the VERIFIED GitHub evidence and score claim/evidence mismatch
     — this is the actual cross-modal fraud signal this dataset is built to
     test, and the one production feature (skill_mismatch / audio_resume_
     mismatch) that every prior run on this project left at 0. Results are
     cached to disk (real_dataset_claims_cache.json) so re-runs don't re-bill
     the API unless a row's content actually changed.
  3. Builds a feature vector per row and evaluates with Stratified 4-fold CV
     (n=24 is too small for a single held-out test split to mean anything).
  4. Reports the same shape of output as train_anomaly_model.py: a rule-vs-
     ML-vs-hybrid ablation, supervised model comparison, feature importance,
     a Gradient Boosting loss curve, and a confusion matrix — but every
     number here is cross-validated out-of-fold on REAL labelled data.

READ THIS BEFORE QUOTING NUMBERS: n=24 is small. Every metric below is
reported with cross-validation, not a single lucky split, but at this size
variance across folds is a real signal, not noise to ignore — report the
spread, not just the mean, in your thesis.
"""
import json
import logging
import re
import time
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report, log_loss,
)

from app.core.config import settings
import app.ml.pipeline_config as cfg
from app.services.llm_client import get_gemini_compatible_schema

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Module-level defaults, overridable via --dataset-path / --results-dir on
# the CLI (see main()) or ML_LABELED_DATASET_PATH / ML_RESULTS_DIR in .env.
DATASET_PATH = cfg.LABELED_DATASET_PATH
RESULTS_DIR = cfg.RESULTS_DIR
CACHE_PATH = RESULTS_DIR / "real_dataset_claims_cache.json"

N_FOLDS = cfg.N_FOLDS  # keep <= the minority class row count

GENERIC_MESSAGE_PATTERNS = [
    r"^initial commit$", r"^update$", r"^fix$", r"^test$",
    r"^wip$", r"^commit\s*\d*$", r"^\.+$",
]


# ----------------------------------------------------------------------
# 1. Parse cached GitHub evidence
# ----------------------------------------------------------------------
def parse_github_evidence(raw: str) -> dict:
    """The column is 'GitHub MSR evidence for github.com/USER: {json}' — strip
    the prefix and parse the JSON payload."""
    brace = raw.find("{")
    return json.loads(raw[brace:])


def generic_message_ratio(messages: List[str]) -> float:
    if not messages:
        return 0.0
    hits = sum(
        1 for m in messages
        if any(re.match(p, m.strip().lower()) for p in GENERIC_MESSAGE_PATTERNS)
    )
    return round(hits / len(messages), 3)


# ----------------------------------------------------------------------
# 2. LLM claim-vs-evidence assessment (the real cross-modal signal)
# ----------------------------------------------------------------------
class ClaimEvidenceAssessment(BaseModel):
    claimed_commit_activity_matches_evidence: bool = Field(
        description="True if spoken claims about commit volume/frequency are consistent with the verified evidence.")
    claimed_experience_matches_resume: bool = Field(
        description="True if spoken experience/seniority claims match what the resume itself states.")
    claimed_tech_stack_matches_evidence: bool = Field(
        description="True if claimed technologies are reasonably supported by the evidence or credibly explained (e.g. private repos).")
    number_of_contradictions: int = Field(
        description="Count of distinct, unresolved contradictions between spoken claims and resume/evidence.")
    mismatch_score: float = Field(
        description="0.0 = fully consistent, 1.0 = severe, multiple unresolved fabrications.")
    key_contradictions: List[str] = Field(
        description="Short description of each unresolved contradiction found. Empty list if none.")


ASSESSMENT_PROMPT = """You are a fraud-detection analyst for technical recruitment. You are given:
1. An interview TRANSCRIPT (what the candidate said, verbatim).
2. The candidate's RESUME text.
3. VERIFIED GitHub evidence (ground truth, independently fetched from the GitHub API — the candidate cannot fake this).

Compare the candidate's spoken and written claims against the verified evidence. Look specifically for:
- Claimed commit volume/frequency that contradicts total_commits / frequency_per_week / average_commit_size_lines.
- Claimed years of experience or seniority that contradicts what the resume itself states.
- Claimed technologies/languages that are absent from primary_languages, without a credible explanation (e.g. private/company repos).
- Claimed pull request, code review, or CI/CD ownership that contradicts total_prs / approval_rate.
- Claimed open-source popularity (stars, followers, forks) that contradicts the evidence.
- Any other claim in the transcript the resume or evidence directly contradicts.

Do NOT penalise the candidate for admitting limitations, correcting themselves, or honestly explaining a discrepancy during the interview — only score contradictions that remain unresolved, or where the candidate asserts something false and does not walk it back.

TRANSCRIPT:
{transcript}

RESUME:
{resume}

VERIFIED GITHUB EVIDENCE (JSON):
{evidence_json}

Return a structured assessment."""


# See app/ml/pipeline_config.py's GEMINI_MODEL comment: gemini-2.5-flash /
# gemini-2.0-flash returned 404 or a 0-request free-tier grant on the
# project's key at time of writing. Change ML_GEMINI_MODEL in .env, or edit
# pipeline_config.py, if your key supports a different model.
GEMINI_MODEL = cfg.GEMINI_MODEL
GEMINI_MIN_INTERVAL_SECONDS = cfg.GEMINI_MIN_INTERVAL_SECONDS
GEMINI_MAX_RETRIES = cfg.GEMINI_MAX_RETRIES


def assess_claims(client: genai.Client, transcript: str, resume: str, evidence: dict) -> ClaimEvidenceAssessment:
    prompt = ASSESSMENT_PROMPT.format(
        transcript=transcript, resume=resume, evidence_json=json.dumps(evidence, indent=2)
    )
    schema = get_gemini_compatible_schema(ClaimEvidenceAssessment)
    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json", response_schema=schema, temperature=0.1
                ),
            )
            return ClaimEvidenceAssessment.model_validate(json.loads(resp.text))
        except genai_errors.ClientError as e:
            is_rate_limit = getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e)
            if is_rate_limit and attempt < GEMINI_MAX_RETRIES - 1:
                wait = 20.0 * (attempt + 1)
                logger.warning(f"Gemini rate limit hit; waiting {wait:.0f}s "
                                f"(attempt {attempt + 1}/{GEMINI_MAX_RETRIES})...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Gemini rate-limit retries exhausted.")


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def row_hash(row) -> str:
    import hashlib
    payload = f"{row['Transcript']}|{row['Resume']}|{row['GitHub_Evidence']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------
# 3. Feature assembly
# ----------------------------------------------------------------------
FEATURE_NAMES = [
    "total_commits", "average_commit_size_lines", "total_prs", "approval_rate",
    "total_repos", "followers", "num_languages", "generic_message_ratio",
    "mismatch_score", "number_of_contradictions",
]


def build_dataset():
    df = pd.read_csv(DATASET_PATH)
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set — required for claim/evidence assessment.")
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    cache = load_cache()

    rows, assessments = [], []
    print("Assessing claim/evidence consistency per row (Gemini, cached)...")
    for i, row in df.iterrows():
        ev = parse_github_evidence(row["GitHub_Evidence"])
        h = row_hash(row)
        cache_key = f"{row['ID']}:{h}"
        if cache_key in cache:
            a = ClaimEvidenceAssessment.model_validate(cache[cache_key])
        else:
            a = assess_claims(client, row["Transcript"], row["Resume"], ev)
            cache[cache_key] = a.model_dump()
            save_cache(cache)
            time.sleep(GEMINI_MIN_INTERVAL_SECONDS)  # stay under the 5 req/min free-tier quota
        assessments.append(a)

        vec = {
            "total_commits": ev.get("total_commits", 0),
            "average_commit_size_lines": ev.get("average_commit_size_lines", 0.0),
            "total_prs": ev.get("total_prs", 0),
            "approval_rate": ev.get("approval_rate", 0.0),
            "total_repos": ev.get("total_repos", 0),
            "followers": ev.get("followers", 0),
            "num_languages": len(ev.get("primary_languages", {}) or {}),
            "generic_message_ratio": generic_message_ratio(ev.get("sample_commit_messages", [])),
            "mismatch_score": a.mismatch_score,
            "number_of_contradictions": a.number_of_contradictions,
        }
        rows.append(vec)
        print(f"  [{i+1}/{len(df)}] {row['Name']:<26} ({row['Case_Type']:<20}) "
              f"mismatch={a.mismatch_score:.2f}  contradictions={a.number_of_contradictions} "
              f"-> Is_Fraud={row['Is_Fraud']}")

    X = pd.DataFrame(rows)[FEATURE_NAMES].to_numpy(dtype=float)
    y = df["Is_Fraud"].to_numpy(dtype=int)
    return X, y, df, assessments


# ----------------------------------------------------------------------
# 4. Reporting helpers (mirrors train_anomaly_model.py's style)
# ----------------------------------------------------------------------
def _metrics(y_true, y_pred, y_score) -> dict:
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4) if len(set(y_true)) > 1 else None,
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4) if len(set(y_true)) > 1 else None,
    }


def _best_threshold(y_true, y_score):
    best = {"threshold": 0.5, "f1": -1.0}
    for t in np.linspace(0.01, 0.99, 99):
        pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best["f1"]:
            best = {
                "threshold": round(float(t), 3), "f1": round(float(f1), 4),
                "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
            }
    return best


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=cfg.LABELED_DATASET_PATH,
                         help=f"Labeled dataset CSV (default {cfg.LABELED_DATASET_PATH}). "
                              "Validate first with: python -m app.ml.dataset_validator --labeled <path>")
    parser.add_argument("--results-dir", type=Path, default=cfg.RESULTS_DIR,
                         help=f"Where to write metrics/figures (default {cfg.RESULTS_DIR}).")
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED,
                         help=f"Random seed for splits and all model fits (default {cfg.RANDOM_SEED}).")
    parser.add_argument("--n-folds", type=int, default=cfg.N_FOLDS,
                         help=f"Stratified CV folds; keep <= minority class count (default {cfg.N_FOLDS}).")
    args = parser.parse_args()

    global DATASET_PATH, RESULTS_DIR, CACHE_PATH, N_FOLDS
    DATASET_PATH = args.dataset_path
    RESULTS_DIR = args.results_dir
    CACHE_PATH = RESULTS_DIR / "real_dataset_claims_cache.json"
    N_FOLDS = args.n_folds
    seed = args.seed

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {}

    X, y, df, assessments = build_dataset()
    print(f"\n{len(y)} real candidates | {int(y.sum())} fraud ({y.mean():.1%}) | {X.shape[1]} features")
    report["dataset"] = {
        "n_total": int(len(y)), "n_fraud": int(y.sum()), "fraud_rate": round(float(y.mean()), 4),
        "n_features": int(X.shape[1]), "feature_names": FEATURE_NAMES,
        "n_unique_people": int(df["GitHub_URL"].nunique()), "real": True,
    }

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    # ------------------------------------------------------------------
    # ABLATION: rule-only (mismatch_score alone) vs ML-only vs hybrid,
    # all evaluated out-of-fold so nothing leaks.
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"ABLATION STUDY  (Stratified {N_FOLDS}-fold CV, out-of-fold, n={len(y)})")
    print("=" * 72)

    rule_score = X[:, FEATURE_NAMES.index("mismatch_score")]
    rule_best = _best_threshold(y, rule_score)

    ml_pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    ml_proba = cross_val_predict(ml_pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    ml_best = _best_threshold(y, ml_proba)

    hybrid_score = 0.5 * rule_score + 0.5 * ml_proba
    hybrid_best = _best_threshold(y, hybrid_score)

    ablation = {
        "rules_only": _metrics(y, (rule_score >= rule_best["threshold"]).astype(int), rule_score),
        "ml_only": _metrics(y, (ml_proba >= ml_best["threshold"]).astype(int), ml_proba),
        "hybrid": _metrics(y, (hybrid_score >= hybrid_best["threshold"]).astype(int), hybrid_score),
    }
    report["ablation"] = ablation
    print(f"{'variant':<14}{'precision':>11}{'recall':>9}{'F1':>9}{'ROC-AUC':>10}{'PR-AUC':>9}")
    print("-" * 72)
    for name, m in ablation.items():
        print(f"{name:<14}{m['precision']:>11.3f}{m['recall']:>9.3f}{m['f1']:>9.3f}"
              f"{(m['roc_auc'] or 0):>10.3f}{(m['pr_auc'] or 0):>9.3f}")
    best_variant = max(ablation.items(), key=lambda kv: kv[1]["f1"])[0]
    print(f"\nBest by F1: {best_variant}")

    # ------------------------------------------------------------------
    # Unsupervised IsolationForest, fit-on-train/score-on-test per fold
    # ------------------------------------------------------------------
    iso_scores = np.zeros(len(y))
    for train_idx, test_idx in cv.split(X, y):
        iso = IsolationForest(contamination=cfg.REAL_ISOLATION_FOREST_CONTAMINATION, random_state=seed)
        iso.fit(X[train_idx])
        iso_scores[test_idx] = -iso.score_samples(X[test_idx])  # higher = more anomalous
    iso_scores = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)
    iso_auc = round(float(roc_auc_score(y, iso_scores)), 4)
    report["isolation_forest_unsupervised"] = {"roc_auc": iso_auc}
    print(f"\nIsolationForest (unsupervised, fit-on-train-fold/score-on-test-fold): "
          f"ROC-AUC vs real label = {iso_auc:.3f}")
    print("  (Unsupervised means it never sees Is_Fraud — this AUC is a diagnostic")
    print("   of how much the anomaly notion alone aligns with real fraud, not a")
    print("   claim that it should be used unsupervised in production.)")

    # ------------------------------------------------------------------
    # Supervised baselines, cross-validated
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"SUPERVISED BASELINES  (Stratified {N_FOLDS}-fold CV, out-of-fold)")
    print("=" * 72)

    baselines = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(
            n_estimators=cfg.REAL_RF_N_ESTIMATORS, max_depth=cfg.REAL_RF_MAX_DEPTH,
            class_weight="balanced", random_state=seed
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=cfg.REAL_GBM_N_ESTIMATORS, learning_rate=cfg.REAL_GBM_LEARNING_RATE,
            max_depth=cfg.REAL_GBM_MAX_DEPTH, random_state=seed
        ),
    }
    report["supervised_baselines"] = {}
    for name, clf in baselines.items():
        proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        best = _best_threshold(y, proba)
        m = _metrics(y, (proba >= best["threshold"]).astype(int), proba)
        report["supervised_baselines"][name] = m
        print(f"{name:<22} CV F1 = {m['f1']:.3f}   ROC-AUC = {(m['roc_auc'] or 0):.3f}   "
              f"(best threshold {best['threshold']})")

    rf_full = RandomForestClassifier(
        n_estimators=cfg.REAL_RF_N_ESTIMATORS, max_depth=cfg.REAL_RF_MAX_DEPTH,
        class_weight="balanced", random_state=seed
    )
    rf_full.fit(X, y)
    importances = sorted(zip(FEATURE_NAMES, rf_full.feature_importances_), key=lambda kv: kv[1], reverse=True)
    report["feature_importance"] = {k: round(float(v), 4) for k, v in importances}
    print(f"\nFeature importance (Random Forest, fit on all {len(y)} rows — exploratory, not stable at this n):")
    for name, imp in importances:
        print(f"  {name:<28}{imp:>6.3f}  {'#' * int(imp * 60)}")

    # ------------------------------------------------------------------
    # Loss curve (Gradient Boosting) — illustrative at n=24
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("TRAINING / VALIDATION LOSS CURVE  (Gradient Boosting)")
    print("=" * 72)
    print(f"NOTE: n={len(y)} gives very few validation points — treat this curve as")
    print("illustrative, not a robust diagnostic, until the labelled set is larger.")

    X_sub_tr, X_val, y_sub_tr, y_val = train_test_split(
        X, y, test_size=cfg.REAL_LOSS_CURVE_TEST_FRACTION, random_state=seed, stratify=y
    )
    loss_clf = GradientBoostingClassifier(
        n_estimators=cfg.REAL_LOSS_CURVE_N_ESTIMATORS, learning_rate=cfg.REAL_LOSS_CURVE_LEARNING_RATE,
        max_depth=cfg.REAL_LOSS_CURVE_MAX_DEPTH, random_state=seed
    )
    loss_clf.fit(X_sub_tr, y_sub_tr)
    train_loss = loss_clf.train_score_
    val_loss = np.array([log_loss(y_val, p, labels=[0, 1]) for p in loss_clf.staged_predict_proba(X_val)])
    iterations = np.arange(1, len(train_loss) + 1)
    best_iter = int(np.argmin(val_loss)) + 1

    report["loss_curve"] = {
        "model": "GradientBoostingClassifier", "n_estimators": len(train_loss),
        "n_train": len(X_sub_tr), "n_val": len(X_val),
        "final_train_loss": round(float(train_loss[-1]), 4),
        "final_val_loss": round(float(val_loss[-1]), 4),
        "best_val_loss": round(float(val_loss.min()), 4), "best_iteration": best_iter,
    }
    print(f"  {len(X_sub_tr)} train / {len(X_val)} validation rows")
    print(f"  Final train loss: {train_loss[-1]:.4f}   Final val loss: {val_loss[-1]:.4f}")
    print(f"  Best val loss:    {val_loss.min():.4f}  at iteration {best_iter}/{len(train_loss)}")

    COLOR_TRAIN, COLOR_VAL, COLOR_GRID, COLOR_AXIS, COLOR_INK, COLOR_MUTED = (
        "#2a78d6", "#eb6834", "#e1e0d9", "#c3c2b7", "#0b0b0b", "#52514e"
    )
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150, facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(iterations, train_loss, color=COLOR_TRAIN, linewidth=2, label="Training loss")
    ax.plot(iterations, val_loss, color=COLOR_VAL, linewidth=2, label="Validation loss")
    ax.axvline(best_iter, color=COLOR_MUTED, linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("Boosting iteration", color=COLOR_INK)
    ax.set_ylabel("Log loss (binomial deviance)", color=COLOR_INK)
    ax.set_title("Real-Data Gradient Boosting — Training vs. Validation Loss (n=24)", color=COLOR_INK, fontsize=11)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_AXIS); ax.spines["bottom"].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_MUTED)
    legend = ax.legend(frameon=False, loc="upper right")
    for t in legend.get_texts():
        t.set_color(COLOR_INK)
    fig.tight_layout()
    loss_plot_path = RESULTS_DIR / "real_dataset_loss_curve.png"
    fig.savefig(loss_plot_path)
    plt.close(fig)

    loss_csv_path = RESULTS_DIR / "real_dataset_loss_curve.csv"
    with open(loss_csv_path, "w") as f:
        f.write("iteration,train_loss,val_loss\n")
        for it, tl, vl in zip(iterations, train_loss, val_loss):
            f.write(f"{it},{tl:.6f},{vl:.6f}\n")

    # ------------------------------------------------------------------
    # Out-of-fold confusion matrix for the winning model (non-leaky)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("OUT-OF-FOLD CONFUSION MATRIX  (hybrid score, best threshold)")
    print("=" * 72)
    hybrid_pred = (hybrid_score >= hybrid_best["threshold"]).astype(int)
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y, hybrid_pred))
    print()
    print(classification_report(y, hybrid_pred, target_names=["genuine", "fraud"], zero_division=0))

    # ------------------------------------------------------------------
    # Persist artefacts
    # ------------------------------------------------------------------
    feat_path = RESULTS_DIR / "real_dataset_features.csv"
    out = pd.DataFrame(X, columns=FEATURE_NAMES)
    out["Is_Fraud"] = y
    out["Case_Type"] = df["Case_Type"].values
    out["Name"] = df["Name"].values
    out["Role"] = df["Role"].values
    out.to_csv(feat_path, index=False)

    metrics_path = RESULTS_DIR / "real_dataset_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    print("ARTEFACTS SAVED")
    print("=" * 72)
    print(f"  metrics      {metrics_path}")
    print(f"  features     {feat_path}")
    print(f"  loss curve   {loss_plot_path}")
    print(f"  loss data    {loss_csv_path}")
    print(f"  claims cache {CACHE_PATH}")
    print(f"\nThis is real, labelled data (n={len(y)}, {df['GitHub_URL'].nunique()} real people).")
    print("Still small: report CV spread, not just the mean, and treat single-fold")
    print("numbers as noisy at this n.")


if __name__ == "__main__":
    main()
