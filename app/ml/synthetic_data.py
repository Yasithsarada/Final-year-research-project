"""
Synthetic candidate generator for the Fraud & Anomaly Detection Model.

DESIGN NOTE — why this generates *profiles*, not feature vectors
----------------------------------------------------------------
An earlier version of this file generated feature vectors directly from
assumed distributions. That is weaker than it looks: it bypasses the real
`FeatureEngineeringLayer`, so the thing being evaluated is not the thing that
runs in production. Any bug in the feature engineering would be invisible to
the experiment.

This version instead simulates raw candidate *behaviour* — weekly commit
series, commit messages, PR outcomes, language spread — assembles genuine
`GitHubProfileMetrics` objects, and pushes them through the **real**
`FeatureEngineeringLayer.compute_fraud_indicators()` and the **real**
`build_feature_vector()`. The experiment therefore exercises the actual
production code path end to end, and the ablation study in
`train_anomaly_model.py` can compare the real rule engine against the ML
model on identical inputs.

IMPORTANT CAVEAT FOR YOUR THESIS
--------------------------------
The *behaviour distributions* below are still assumptions I encoded, not
observations. Results computed on this data are a pipeline sanity check and
a lower bound — NOT an empirical finding. Do not present them as your result.

To produce a defensible result, replace `generate_profile_dataset()` with real
data (see FRAUD_DETECTION_GUIDANCE.md §4):
  • Genuine class: real public GitHub profiles run through MSRAgent.
  • Fraud class: throwaway accounts where you scripted the padding behaviour
    yourself, so labels are correct by construction.
Nothing downstream needs to change — the training script consumes profiles.
"""
from typing import List, Tuple

import numpy as np

from app.models.github_metrics import (
    GitHubProfileMetrics,
    CommitHistory,
    PullRequestMetrics,
    RepositoryMetadata,
)

__all__ = [
    "generate_profile_dataset",
    "build_xy",
    "GENUINE_MESSAGES",
    "GENERIC_MESSAGES",
]

GENUINE_MESSAGES = [
    "Add JWT refresh token rotation",
    "Fix race condition in payment webhook handler",
    "Refactor user serializer to drop N+1 query",
    "Bump sqlalchemy to 2.0.30 and migrate legacy Query API",
    "Handle timezone-naive datetimes in report export",
    "Add integration test for checkout retry path",
    "Cache repo language lookup to cut API calls",
    "Correct off-by-one in pagination offset",
]

GENERIC_MESSAGES = ["update", "fix", "wip", "test", "commit", "initial commit", "."]

LANGUAGE_POOL = ["Python", "JavaScript", "TypeScript", "Java", "Go", "C++", "Ruby", "Rust"]


def _languages(rng: np.random.Generator, n: int) -> dict:
    n = int(np.clip(n, 1, len(LANGUAGE_POOL)))
    picked = rng.choice(LANGUAGE_POOL, size=n, replace=False)
    return {str(lang): int(rng.integers(1, 8)) for lang in picked}


def _messages(rng: np.random.Generator, generic_ratio: float, k: int = 20) -> List[str]:
    n_generic = int(round(k * generic_ratio))
    out = [str(rng.choice(GENERIC_MESSAGES)) for _ in range(n_generic)]
    out += [str(rng.choice(GENUINE_MESSAGES)) for _ in range(k - n_generic)]
    # Shuffle via an index permutation rather than rng.shuffle(list), which
    # behaves inconsistently across numpy versions for Python lists.
    order = rng.permutation(len(out))
    return [out[i] for i in order]


def _genuine_profile(rng: np.random.Generator) -> Tuple[GitHubProfileMetrics, float, float]:
    """A real developer: sustained activity over months/years, modest commit
    sizes, descriptive messages, healthy PR acceptance.

    Three archetypes are mixed. Validating this pipeline against a real
    long-tenured GitHub account (19 years, 997 observed weeks, only 87 active)
    showed commit_coefficient_variation of ~4.9 — far above the ~0.08-0.25
    range this generator previously produced for ANY genuine profile, and
    squarely inside the synthetic fraud range. That happened because only
    short-tenure (<=160 week), steady-cadence profiles were being generated.
    Real veteran contributors are only active during specific projects/
    releases with long genuine silences between — just as "bursty" by this
    metric as a padded profile. The "veteran" archetype reproduces that.

    The "crunch" archetype (added after a supervisor review flagged the
    original benchmark as unrealistically separable) covers a genuine but
    real pattern — a legitimate developer working an intense pre-deadline
    sprint, whose commit history is structurally similar to fraud's "burst"
    subtype (long quiet stretch, then a concentrated spike). Distinguishing
    real crunch periods from padding cannot rely on burst shape alone; a
    model that generalises has to weigh the other features too, which is the
    point of including it.
    """
    archetype_roll = rng.random()
    if archetype_roll < 0.30:
        # Veteran: long history, sparse bursts of real project work.
        n_weeks = int(rng.integers(200, 1000))
        weeks_arr = np.zeros(n_weeks, dtype=int)
        n_bursts = int(rng.integers(2, 8))
        for _ in range(n_bursts):
            start = int(rng.integers(0, n_weeks))
            length = int(rng.integers(1, 6))
            rate = rng.uniform(2, 20)
            seg = rng.poisson(rate, min(length, n_weeks - start))
            weeks_arr[start:start + len(seg)] += seg
        weeks = [int(max(0, w)) for w in weeks_arr.tolist()]
    elif archetype_roll < 0.45:
        # Crunch: genuine, but a single concentrated pre-deadline sprint —
        # structurally close to fraud's "burst" subtype on purpose.
        n_weeks = int(rng.integers(30, 120))
        weeks_arr = np.zeros(n_weeks, dtype=int)
        n_active = int(rng.integers(1, 3))
        for i in range(n_active):
            weeks_arr[-(i + 1)] = int(rng.integers(25, 150))
        # A little background activity elsewhere in the window, unlike fraud's
        # near-total silence — the one structural difference left.
        for _ in range(int(rng.integers(2, 6))):
            idx = int(rng.integers(0, max(n_weeks - n_active, 1)))
            weeks_arr[idx] += int(rng.integers(1, 5))
        weeks = [int(max(0, w)) for w in weeks_arr.tolist()]
    else:
        n_weeks = int(rng.integers(40, 160))
        base_rate = rng.uniform(1.5, 6.0)
        weeks_arr = rng.poisson(base_rate, n_weeks).astype(int)

        # Real people take holidays and get busy at work — insert quiet stretches.
        for _ in range(int(rng.integers(0, 4))):
            start = int(rng.integers(0, max(n_weeks - 4, 1)))
            weeks_arr[start:start + int(rng.integers(2, 6))] = 0

        weeks = [int(max(0, w)) for w in weeks_arr.tolist()]

    if sum(weeks) == 0:
        weeks[0] = 1

    # Widened from beta(2,8)/normal(0.82,0.12) — the original distributions
    # barely overlapped with the fraud class on these two features, making
    # them near-perfect linear separators. Real genuine candidates sometimes
    # write terse commit messages or have a rough PR history too.
    generic_ratio = float(np.clip(rng.beta(2, 6), 0, 1))
    total_prs = int(rng.integers(4, 40))
    approval = float(np.clip(rng.normal(0.78, 0.15), 0.0, 1.0))

    gh = GitHubProfileMetrics(
        username=f"genuine_{int(rng.integers(1_000_000))}",
        commit_history=CommitHistory(
            total_commits=sum(weeks),
            frequency_per_week=round(sum(weeks) / max(len([w for w in weeks if w > 0]), 1), 2),
            average_commit_size_lines=float(np.clip(rng.normal(110, 60), 5, 900)),
            merging_percentage=float(np.clip(rng.normal(12, 6), 0, 60)),
            commit_messages=_messages(rng, generic_ratio),
            weekly_commit_counts=weeks,
        ),
        pull_requests=PullRequestMetrics(
            total_prs=total_prs,
            accepted_prs=int(total_prs * approval),
            rejected_prs=total_prs - int(total_prs * approval),
            approval_rate=round(approval, 3),
            average_comments_per_pr=float(np.clip(rng.normal(3.5, 1.5), 0, 15)),
        ),
        repository_metadata=RepositoryMetadata(
            total_repos=int(rng.integers(4, 30)),
            total_stars=int(rng.integers(0, 300)),
            total_forks=int(rng.integers(0, 80)),
            primary_languages=_languages(rng, int(rng.integers(2, 7))),
            followers=int(rng.integers(2, 250)),
            collaborations=int(rng.integers(4, 30)),
        ),
    )
    # Widened from beta(2,10) — see generic_ratio/approval note above. A
    # genuine candidate occasionally has a skill they haven't demonstrated
    # publicly yet, or misremembers a detail on a call; that shouldn't be
    # (and structurally now isn't) indistinguishable from zero mismatch.
    skill_mismatch = float(np.clip(rng.beta(2, 7), 0, 1))
    audio_mismatch = float(np.clip(rng.beta(2, 7), 0, 1))
    return gh, skill_mismatch, audio_mismatch


def _fraud_profile(rng: np.random.Generator) -> Tuple[GitHubProfileMetrics, float, float]:
    """A padded profile. Four sub-types are mixed so the class is not
    trivially separable by any single feature — which is the realistic case
    and prevents the model from looking better than it is."""
    subtype = rng.choice(
        ["burst", "bulk_dump", "slow_padding", "disguised"], p=[0.38, 0.27, 0.20, 0.15]
    )
    n_weeks = int(rng.integers(30, 120))
    weeks = [0] * n_weeks

    if subtype == "burst":
        # Long dormancy, then a spike just before applying. Range brought
        # down from 40-250 so it overlaps the genuine "crunch" archetype's
        # 25-150 range instead of always dwarfing it.
        n_active = int(rng.integers(1, 4))
        for i in range(n_active):
            weeks[-(i + 1)] = int(rng.integers(25, 180))
        avg_size = float(np.clip(rng.normal(220, 120), 20, 2000))
        generic_ratio = float(np.clip(rng.beta(4, 4), 0, 1))

    elif subtype == "bulk_dump":
        # A handful of enormous commits — someone else's codebase pushed up.
        for i in range(int(rng.integers(2, 6))):
            weeks[int(rng.integers(0, n_weeks))] = int(rng.integers(1, 5))
        avg_size = float(np.clip(rng.normal(3500, 1500), 800, 12000))
        generic_ratio = float(np.clip(rng.beta(3, 4), 0, 1))

    elif subtype == "slow_padding":  # the hardest case to catch, deliberately included
        base = rng.uniform(0.3, 1.2)
        weeks = [int(max(0, w)) for w in rng.poisson(base, n_weeks)]
        spike = int(rng.integers(0, n_weeks))
        weeks[spike] = int(rng.integers(25, 90))
        avg_size = float(np.clip(rng.normal(400, 250), 30, 3000))
        generic_ratio = float(np.clip(rng.beta(4, 5), 0, 1))

    else:  # disguised — commit pattern looks like a genuine steady contributor;
        # the tell is elsewhere (skill/claim mismatch, PR history), not in the
        # weekly shape. Represents padding sophisticated enough to not trip
        # the burst/CV features at all.
        base_rate = rng.uniform(1.0, 4.0)
        weeks_arr = rng.poisson(base_rate, n_weeks).astype(int)
        weeks = [int(max(0, w)) for w in weeks_arr.tolist()]
        avg_size = float(np.clip(rng.normal(150, 90), 10, 900))
        generic_ratio = float(np.clip(rng.beta(3, 5), 0, 1))

    if sum(weeks) == 0:
        weeks[-1] = int(rng.integers(30, 120))

    total_prs = int(rng.integers(0, 15))
    # Widened from normal(0.42,0.22) so the lower tail of genuine approval
    # (now normal(0.75,0.18)) and the upper tail here genuinely overlap.
    approval = float(np.clip(rng.normal(0.45, 0.22), 0.0, 1.0))

    gh = GitHubProfileMetrics(
        username=f"fraud_{int(rng.integers(1_000_000))}",
        commit_history=CommitHistory(
            total_commits=sum(weeks),
            frequency_per_week=round(sum(weeks) / max(len([w for w in weeks if w > 0]), 1), 2),
            average_commit_size_lines=avg_size,
            merging_percentage=float(np.clip(rng.normal(3, 3), 0, 40)),
            commit_messages=_messages(rng, generic_ratio),
            weekly_commit_counts=weeks,
        ),
        pull_requests=PullRequestMetrics(
            total_prs=total_prs,
            accepted_prs=int(total_prs * approval),
            rejected_prs=total_prs - int(total_prs * approval),
            average_comments_per_pr=float(np.clip(rng.normal(1.0, 0.8), 0, 10)),
            approval_rate=round(approval, 3),
        ),
        repository_metadata=RepositoryMetadata(
            total_repos=int(rng.integers(1, 12)),
            total_stars=int(rng.integers(0, 15)),
            total_forks=int(rng.integers(0, 5)),
            primary_languages=_languages(rng, int(rng.integers(1, 4))),
            followers=int(rng.integers(0, 25)),
            collaborations=int(rng.integers(1, 12)),
        ),
    )
    # Padded candidates tend to over-claim relative to evidence, but not
    # always — narrowed from beta(5,3)/beta(4,3) (mean 0.625/0.571) to
    # overlap with the widened genuine range (mean 0.25) instead of sitting
    # almost entirely above it.
    skill_mismatch = float(np.clip(rng.beta(4, 4), 0, 1))
    audio_mismatch = float(np.clip(rng.beta(4, 4), 0, 1))
    return gh, skill_mismatch, audio_mismatch


def generate_profile_dataset(
    n_genuine: int = 700,
    n_fraud: int = 130,
    seed: int = 42,
):
    """Returns (profiles, skill_mismatches, audio_mismatches, labels).

    labels: 1 = fraudulent/padded, 0 = genuine.
    ~15.7% fraud rate, matching the minority-class assumption behind
    IsolationForest's `contamination` parameter.
    """
    rng = np.random.default_rng(seed)
    profiles, skills, audios, labels = [], [], [], []

    for _ in range(n_genuine):
        gh, s, a = _genuine_profile(rng)
        profiles.append(gh); skills.append(s); audios.append(a); labels.append(0)

    for _ in range(n_fraud):
        gh, s, a = _fraud_profile(rng)
        profiles.append(gh); skills.append(s); audios.append(a); labels.append(1)

    idx = rng.permutation(len(profiles))
    return (
        [profiles[i] for i in idx],
        np.array(skills)[idx],
        np.array(audios)[idx],
        np.array(labels)[idx],
    )


def build_xy(profiles, skills, audios):
    """Runs the REAL FeatureEngineeringLayer + build_feature_vector over the
    generated profiles, so the experiment evaluates the production code path.

    Returns (X, rule_scores) where rule_scores are the rule-based engine's own
    anomaly scores — used for the ablation study.
    """
    # Imported here to keep this module importable without the service layer.
    from app.services.feature_engineering import FeatureEngineeringLayer
    from app.services.anomaly_model import build_feature_vector

    fel = FeatureEngineeringLayer()
    X, rule_scores = [], []
    for gh, s, a in zip(profiles, skills, audios):
        fi = fel.compute_fraud_indicators(gh)
        X.append(build_feature_vector(gh, fi, float(s), float(a)))
        rule_scores.append(fi.anomaly_score)
    return np.vstack(X), np.array(rule_scores)
