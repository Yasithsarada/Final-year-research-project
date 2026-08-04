import logging
from pathlib import Path
from typing import Optional

import numpy as np

# scikit-learn / joblib are guarded imports. If they are not installed the
# rest of the API (resume parsing, CCS, MSR) must still boot — the ML
# anomaly detector simply degrades to a neutral score and the rule-based
# fraud indicators carry the Decision Engine on their own.
try:
    import joblib
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    joblib = None
    IsolationForest = None
    SKLEARN_AVAILABLE = False

from app.core.config import BASE_DIR
from app.models.evaluation import FraudIndicators
from app.models.github_metrics import GitHubProfileMetrics

logger = logging.getLogger(__name__)

# Fixed feature order — training (app/ml/train_anomaly_model.py) and
# inference (this module) MUST agree on this ordering, so it lives in one
# place and both sides import it.
FEATURE_NAMES = [
    "calendar_commit_frequency",       # sustained commits/week over full history (capped, normalised)
    "avg_commit_size_norm",            # average lines changed per commit (capped, normalised)
    "burst_concentration_ratio",       # share of commits packed into the busiest 10% of active weeks
    "commit_coefficient_variation",    # std/mean of the weekly commit series (bursty vs steady)
    "generic_message_ratio",           # fraction of generic/auto-generated commit messages
    "pr_rejection_rate",               # 1 - PR approval rate
    "language_diversity_norm",         # distinct languages used (capped, normalised)
    "skill_mismatch",                  # claimed skills not evidenced in GitHub
    "audio_resume_mismatch",           # interview claims vs resume inconsistency
]

DEFAULT_MODEL_PATH = BASE_DIR / "app_data" / "models" / "isolation_forest.joblib"


def build_feature_vector(
    gh: GitHubProfileMetrics,
    fi: FraudIndicators,
    skill_mismatch: float = 0.0,
    audio_resume_mismatch: float = 0.0,
) -> np.ndarray:
    """Builds the fixed-order numeric feature vector consumed by the ML
    anomaly detector, from the same evidence the rule-based Fraud Indicators
    already computed (see FeatureEngineeringLayer.compute_fraud_indicators).

    Keeping vector construction in one shared function guarantees training
    data and live inference are built identically.
    """
    ch = gh.commit_history
    weeks = ch.weekly_commit_counts
    calendar_freq = ch.total_commits / len(weeks) if weeks else ch.frequency_per_week

    vec = [
        min(calendar_freq / 10.0, 1.0),
        min(ch.average_commit_size_lines / 500.0, 1.0),
        fi.burst_concentration_ratio,
        min(fi.commit_coefficient_variation / 5.0, 1.0),
        fi.generic_message_ratio,
        fi.pr_rejection_rate,
        min(len(gh.repository_metadata.primary_languages) / 6.0, 1.0),
        skill_mismatch,
        audio_resume_mismatch,
    ]
    return np.array(vec, dtype=float)


class MLAnomalyDetector:
    """Wraps a scikit-learn IsolationForest to provide a genuine
    machine-learning anomaly score for the Fraud & Anomaly Detection Model,
    as an unsupervised complement to the deterministic rule-based flags in
    FeatureEngineeringLayer.compute_fraud_indicators().

    Why IsolationForest / unsupervised: at this stage of the project there
    is no labelled set of confirmed-fraudulent real candidates to train a
    supervised classifier on. IsolationForest only needs a "mostly genuine"
    reference population and isolates points that are structurally
    different from it — which is exactly the "risk indicators rather than
    definitive judgements" framing used in the proposal. Once real labelled
    outcomes exist (e.g. from recruiter feedback), this can be swapped for
    a supervised model (e.g. gradient boosting) without touching the rest
    of the Decision Engine, since it only depends on `score()`.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self._model = None
        self._score_min: float = -0.5
        self._score_max: float = 0.5
        self._load()

    def _load(self) -> None:
        if not SKLEARN_AVAILABLE:
            logger.warning(
                "MLAnomalyDetector: scikit-learn/joblib not installed. "
                "Run `pip install scikit-learn joblib`. Falling back to a "
                "neutral 0.5 anomaly score."
            )
            return
        if self.model_path.exists():
            try:
                bundle = joblib.load(self.model_path)
                self._model = bundle["model"]
                self._score_min = bundle.get("score_min", self._score_min)
                self._score_max = bundle.get("score_max", self._score_max)
                logger.info(f"MLAnomalyDetector: loaded model from {self.model_path}")
            except Exception as e:
                logger.warning(f"MLAnomalyDetector: failed to load model ({e}); running unfitted.")
                self._model = None
        else:
            logger.warning(
                f"MLAnomalyDetector: no trained model found at {self.model_path}. "
                "Run `python -m app.ml.train_anomaly_model` from the project root "
                "to train one. Falling back to a neutral 0.5 score until then."
            )

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def score(self, feature_vector: np.ndarray) -> float:
        """Returns a normalised anomaly score in [0, 1]; 1.0 = most anomalous."""
        if self._model is None:
            return 0.5  # neutral fallback so the pipeline still runs pre-training
        raw = self._model.decision_function(feature_vector.reshape(1, -1))[0]
        # IsolationForest.decision_function: higher = more "normal". We flip
        # and min-max normalise using the range observed at training time.
        span = max(self._score_max - self._score_min, 1e-6)
        normalised = (self._score_max - raw) / span
        return float(np.clip(normalised, 0.0, 1.0))

    def predict_is_anomaly(self, X: np.ndarray) -> np.ndarray:
        """Hard binary labels (1 = anomaly) using IsolationForest's own
        learned decision boundary, rather than an arbitrary cut on the
        normalised score. Used for precision/recall reporting.
        """
        if self._model is None:
            return np.zeros(len(X), dtype=int)
        # sklearn returns -1 for outliers, +1 for inliers
        return (self._model.predict(X) == -1).astype(int)

    def fit(self, X: np.ndarray, contamination: float = 0.15, random_state: int = 42) -> "MLAnomalyDetector":
        if not SKLEARN_AVAILABLE:
            raise RuntimeError(
                "scikit-learn is required to train the anomaly model. "
                "Install it with: pip install scikit-learn joblib"
            )
        contamination = float(np.clip(contamination, 0.01, 0.5))
        self._model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
        )
        self._model.fit(X)
        scores = self._model.decision_function(X)
        self._score_min, self._score_max = float(scores.min()), float(scores.max())
        return self

    def save(self, path: Optional[Path] = None) -> None:
        path = path or self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self._model, "score_min": self._score_min, "score_max": self._score_max},
            path,
        )
        logger.info(f"MLAnomalyDetector: model saved to {path}")
