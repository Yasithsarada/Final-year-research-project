"""
Single source of truth for every path, seed, and hyperparameter used across
app/ml/*.py. Change a dataset location or a hyperparameter here — or
override it per-run — rather than editing the training/evaluation scripts.

THREE WAYS TO OVERRIDE A VALUE, checked in this order (highest wins):
  1. A CLI flag on the specific script you're running, e.g.:
         python -m app.ml.train_on_real_dataset --dataset-path my_data.csv
     Run any script with --help to see what it accepts.
  2. An environment variable / .env entry, prefixed ML_, e.g. in .env:
         ML_RANDOM_SEED=7
         ML_LABELED_DATASET_PATH=datasets/my_other_dataset.csv
  3. The default value below — just edit this file directly.

This file has NO side effects beyond creating the two output directories, so
it is always safe to `import app.ml.pipeline_config as cfg`.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

from app.core.config import BASE_DIR

load_dotenv(BASE_DIR / ".env")


def _env(name: str, default, cast=str):
    val = os.environ.get(f"ML_{name}")
    if val is None:
        return default
    return cast(val)


# ---------------------------------------------------------------------------
# PATHS — change these to point at different datasets or output locations.
# ---------------------------------------------------------------------------

# The real, human-labelled dataset with an Is_Fraud column. Required columns
# are documented in app/ml/dataset_validator.py and app/ml/README.md.
LABELED_DATASET_PATH = Path(_env(
    "LABELED_DATASET_PATH",
    str(BASE_DIR / "datasets" / "interview_cv_github_fraud_dataset.csv"),
))

# A real dataset WITHOUT a fraud label — used by predict_unlabeled_profiles.py
# to score new candidates with the classifier trained on LABELED_DATASET_PATH.
UNLABELED_DATASET_PATH = Path(_env(
    "UNLABELED_DATASET_PATH",
    str(BASE_DIR / "datasets" / "transcripted interview data-filtered.csv"),
))

# Where every script writes its metrics/figures/caches.
RESULTS_DIR = Path(_env("RESULTS_DIR", str(BASE_DIR / "app_data" / "ml_results")))

# Where the persisted, production-loaded model lives.
MODEL_DIR = Path(_env("MODEL_DIR", str(BASE_DIR / "app_data" / "models")))
ISOLATION_FOREST_PATH = MODEL_DIR / "isolation_forest.joblib"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------------
# Every train/test split, every model's random_state, and the synthetic
# generator's RNG all derive from this single seed. Same seed + same dataset
# => byte-identical splits and near-identical trained models (sklearn's own
# floating-point/threading nondeterminism aside — set PYTHONHASHSEED and use
# single-threaded fits if you need bit-exact reproduction across machines).
RANDOM_SEED = _env("RANDOM_SEED", 42, int)

# ---------------------------------------------------------------------------
# SYNTHETIC DATA GENERATION (train_anomaly_model.py, app/ml/synthetic_data.py)
# ---------------------------------------------------------------------------
SYNTHETIC_N_GENUINE = _env("SYNTHETIC_N_GENUINE", 700, int)
SYNTHETIC_N_FRAUD = _env("SYNTHETIC_N_FRAUD", 130, int)
TRAIN_TEST_SPLIT = _env("TRAIN_TEST_SPLIT", 0.30, float)

# ---------------------------------------------------------------------------
# HYBRID RULE/ML WEIGHTING
# ---------------------------------------------------------------------------
# Mirrors DecisionEngine's blend of the rule-based score and the ML score,
# used when computing the "hybrid" variant in the ablation study.
HYBRID_RULE_WEIGHT = _env("HYBRID_RULE_WEIGHT", 0.50, float)
HYBRID_ML_WEIGHT = _env("HYBRID_ML_WEIGHT", 0.50, float)

# ---------------------------------------------------------------------------
# SUPERVISED BASELINE HYPERPARAMETERS (reference models, not production)
# ---------------------------------------------------------------------------
RF_N_ESTIMATORS = _env("RF_N_ESTIMATORS", 300, int)
GBM_N_ESTIMATORS = _env("GBM_N_ESTIMATORS", 300, int)
GBM_LEARNING_RATE = _env("GBM_LEARNING_RATE", 0.05, float)
GBM_MAX_DEPTH = _env("GBM_MAX_DEPTH", 3, int)

# Loss-curve diagnostic model — deliberately higher capacity than GBM_* above
# so a real overfitting signature (val loss rising after some iteration) is
# visible rather than masked by training loss decaying toward zero.
LOSS_CURVE_N_ESTIMATORS = _env("LOSS_CURVE_N_ESTIMATORS", 200, int)
LOSS_CURVE_LEARNING_RATE = _env("LOSS_CURVE_LEARNING_RATE", 0.12, float)
LOSS_CURVE_MAX_DEPTH = _env("LOSS_CURVE_MAX_DEPTH", 6, int)
LOSS_CURVE_VAL_FRACTION = _env("LOSS_CURVE_VAL_FRACTION", 0.20, float)

# ---------------------------------------------------------------------------
# REAL-DATA CROSS-VALIDATION AND MODEL HYPERPARAMETERS (train_on_real_dataset.py)
# ---------------------------------------------------------------------------
# Number of stratified folds for out-of-fold evaluation on the labelled
# dataset. Keep this <= the minority class count (currently 15 fraud rows).
N_FOLDS = _env("N_FOLDS", 4, int)

# Shallower than the synthetic SYNTHETIC_*/RF_*/GBM_* settings above —
# deliberately, because n=34 overfits fast with the synthetic script's
# deeper/larger models. Increase these once the real labelled dataset is
# meaningfully larger (see app/ml/README.md "Growing the real dataset").
REAL_ISOLATION_FOREST_CONTAMINATION = _env("REAL_ISOLATION_FOREST_CONTAMINATION", 0.5, float)
REAL_RF_N_ESTIMATORS = _env("REAL_RF_N_ESTIMATORS", 200, int)
REAL_RF_MAX_DEPTH = _env("REAL_RF_MAX_DEPTH", 4, int)
REAL_GBM_N_ESTIMATORS = _env("REAL_GBM_N_ESTIMATORS", 100, int)
REAL_GBM_LEARNING_RATE = _env("REAL_GBM_LEARNING_RATE", 0.05, float)
REAL_GBM_MAX_DEPTH = _env("REAL_GBM_MAX_DEPTH", 2, int)
REAL_LOSS_CURVE_N_ESTIMATORS = _env("REAL_LOSS_CURVE_N_ESTIMATORS", 150, int)
REAL_LOSS_CURVE_LEARNING_RATE = _env("REAL_LOSS_CURVE_LEARNING_RATE", 0.05, float)
REAL_LOSS_CURVE_MAX_DEPTH = _env("REAL_LOSS_CURVE_MAX_DEPTH", 2, int)
REAL_LOSS_CURVE_TEST_FRACTION = _env("REAL_LOSS_CURVE_TEST_FRACTION", 0.25, float)

# ---------------------------------------------------------------------------
# GEMINI LLM CLAIM/EVIDENCE ASSESSMENT
# ---------------------------------------------------------------------------
# gemini-2.5-flash / gemini-2.0-flash returned 404 or a 0-request free-tier
# grant on the project's key at time of writing; gemini-flash-lite-latest is
# what actually has quota. If you have a different/paid key, you can likely
# use a stronger model — change this value and re-run, the whole pipeline is
# indifferent to which Gemini model produces the assessment.
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-flash-lite-latest")
GEMINI_MIN_INTERVAL_SECONDS = _env("GEMINI_MIN_INTERVAL_SECONDS", 13.0, float)
GEMINI_MAX_RETRIES = _env("GEMINI_MAX_RETRIES", 5, int)

# ---------------------------------------------------------------------------
# LIVE GITHUB EVIDENCE FETCH (predict_unlabeled_profiles.py)
# ---------------------------------------------------------------------------
MAX_CONCURRENT_GITHUB_FETCHES = _env("MAX_CONCURRENT_GITHUB_FETCHES", 6, int)

# GitHub's /stats/* endpoints compute asynchronously and answer 202 with an
# empty body until the cache warms — see app/services/msr_agent.py.
STATS_MAX_RETRIES = _env("STATS_MAX_RETRIES", 6, int)
STATS_RETRY_DELAY_SECONDS = _env("STATS_RETRY_DELAY_SECONDS", 3.0, float)

# ---------------------------------------------------------------------------
# NO-EVIDENCE FALLBACK (candidates with an unresolvable/empty GitHub account)
# ---------------------------------------------------------------------------
NO_EVIDENCE_RISK_THRESHOLD = _env("NO_EVIDENCE_RISK_THRESHOLD", 0.30, float)


def summary() -> str:
    """Human-readable dump of every effective setting — run this file
    directly (`python -m app.ml.pipeline_config`) to sanity-check what a
    script will actually use before a long run."""
    lines = ["Effective ML pipeline configuration:", ""]
    for name, val in sorted(globals().items()):
        if name.isupper():
            lines.append(f"  {name:32} = {val}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
