"""
Unit tests for the Fraud & Anomaly Detection module.

Run from the `reasearch v2` project root:
    python -m unittest app.tests.test_fraud_detection -v

These tests are deliberately written around the *behaviour that was broken*
before, so they double as regression tests. In particular
`test_burst_vs_steady_are_distinguishable` is the key one: under the previous
implementation a steady contributor and a burst contributor with the same
commit total produced identical feature values, so the burst detector could
not fire. If that test ever fails again, the detector has regressed to being
blind.
"""
import unittest

from app.models.github_metrics import (
    GitHubProfileMetrics,
    CommitHistory,
    PullRequestMetrics,
    RepositoryMetadata,
)
from app.services.feature_engineering import FeatureEngineeringLayer
from app.services.anomaly_model import build_feature_vector, FEATURE_NAMES


def make_profile(
    weekly_counts,
    avg_commit_size=80.0,
    messages=None,
    total_prs=10,
    approval_rate=0.8,
    languages=None,
) -> GitHubProfileMetrics:
    """Builds a GitHubProfileMetrics fixture from a weekly commit series."""
    total = sum(weekly_counts)
    active = [c for c in weekly_counts if c > 0]
    freq = total / max(len(active), 1)
    return GitHubProfileMetrics(
        username="testuser",
        commit_history=CommitHistory(
            total_commits=total,
            frequency_per_week=round(freq, 2),
            average_commit_size_lines=avg_commit_size,
            merging_percentage=10.0,
            commit_messages=messages if messages is not None else ["Add user auth flow"] * 10,
            weekly_commit_counts=weekly_counts,
        ),
        pull_requests=PullRequestMetrics(
            total_prs=total_prs,
            accepted_prs=int(total_prs * approval_rate),
            rejected_prs=total_prs - int(total_prs * approval_rate),
            approval_rate=approval_rate,
            average_comments_per_pr=3.0,
        ),
        repository_metadata=RepositoryMetadata(
            total_repos=8,
            total_stars=25,
            total_forks=5,
            primary_languages=languages if languages is not None else {"Python": 5, "JavaScript": 3},
            followers=20,
            collaborations=8,
        ),
    )


class TestBurstDetection(unittest.TestCase):
    def setUp(self):
        self.fel = FeatureEngineeringLayer()

    def test_steady_contributor_is_not_flagged(self):
        """Two years of ~3 commits/week should look completely normal."""
        weeks = [3] * 104
        fi = self.fel.compute_fraud_indicators(make_profile(weeks))

        self.assertFalse(fi.burst_commits_detected)
        self.assertLess(fi.burst_concentration_ratio, 0.10)
        # Perfectly steady series => near-zero variability
        self.assertLess(fi.commit_coefficient_variation, 0.10)

    def test_burst_contributor_is_flagged(self):
        """A year of silence then a single 300-commit week is the classic
        profile-padding pattern described in the proposal."""
        weeks = [0] * 51 + [300]
        fi = self.fel.compute_fraud_indicators(make_profile(weeks))

        self.assertTrue(fi.burst_commits_detected)
        self.assertGreater(fi.burst_concentration_ratio, 0.90)
        self.assertGreater(fi.anomaly_score, 0.0)

    def test_burst_vs_steady_are_distinguishable(self):
        """REGRESSION TEST for the original bug.

        Both profiles have exactly 300 total commits. Previously they produced
        identical features because zero-weeks were discarded and the burst
        heuristic algebraically cancelled out. They must now differ sharply.
        """
        steady = self.fel.compute_fraud_indicators(make_profile([5] * 60))   # 300 commits
        bursty = self.fel.compute_fraud_indicators(make_profile([0] * 59 + [300]))

        self.assertEqual(sum([5] * 60), sum([0] * 59 + [300]))  # same total
        self.assertNotEqual(steady.burst_concentration_ratio, bursty.burst_concentration_ratio)
        self.assertGreater(bursty.burst_concentration_ratio, steady.burst_concentration_ratio)
        self.assertFalse(steady.burst_commits_detected)
        self.assertTrue(bursty.burst_commits_detected)

    def test_short_history_does_not_crash(self):
        """Sparse/new accounts must degrade gracefully, not raise."""
        for weeks in ([], [1], [0, 0], [2, 3]):
            fi = self.fel.compute_fraud_indicators(make_profile(weeks))
            self.assertGreaterEqual(fi.anomaly_score, 0.0)
            self.assertLessEqual(fi.anomaly_score, 1.0)

    def test_generic_commit_messages_flagged(self):
        fi = self.fel.compute_fraud_indicators(
            make_profile([3] * 20, messages=["update", "fix", "wip", "update", "test"])
        )
        self.assertGreater(fi.generic_message_ratio, 0.60)
        self.assertTrue(any("commit messages" in f.lower() for f in fi.flags))

    def test_bulk_dump_flagged_as_fake_activity(self):
        """One huge push of copied code, no sustained history."""
        fi = self.fel.compute_fraud_indicators(
            make_profile([0] * 40 + [3], avg_commit_size=4000.0)
        )
        self.assertTrue(fi.fake_activity_detected)

    def test_anomaly_score_stays_in_range(self):
        """Worst case on every dimension must still clamp to <= 1.0."""
        fi = self.fel.compute_fraud_indicators(
            make_profile(
                [0] * 80 + [500],
                avg_commit_size=9000.0,
                messages=["update"] * 10,
                total_prs=30,
                approval_rate=0.05,
            )
        )
        self.assertLessEqual(fi.anomaly_score, 1.0)
        self.assertGreaterEqual(fi.anomaly_score, 0.0)


class TestFeatureVector(unittest.TestCase):
    def setUp(self):
        self.fel = FeatureEngineeringLayer()

    def test_vector_length_matches_declared_feature_names(self):
        """Guards against training/inference drifting out of sync."""
        gh = make_profile([4] * 30)
        fi = self.fel.compute_fraud_indicators(gh)
        vec = build_feature_vector(gh, fi)
        self.assertEqual(len(vec), len(FEATURE_NAMES))

    def test_vector_values_are_normalised(self):
        """Every feature must land in [0, 1] so no single one dominates."""
        gh = make_profile([0] * 50 + [400], avg_commit_size=9000.0)
        fi = self.fel.compute_fraud_indicators(gh)
        vec = build_feature_vector(gh, fi, skill_mismatch=1.0, audio_resume_mismatch=1.0)
        for name, value in zip(FEATURE_NAMES, vec):
            self.assertGreaterEqual(value, 0.0, f"{name} below 0")
            self.assertLessEqual(value, 1.0, f"{name} above 1")

    def test_burst_profile_scores_higher_than_steady(self):
        """End-to-end sanity: the vector must preserve the burst signal."""
        steady = make_profile([5] * 60)
        bursty = make_profile([0] * 59 + [300])
        v_steady = build_feature_vector(steady, self.fel.compute_fraud_indicators(steady))
        v_bursty = build_feature_vector(bursty, self.fel.compute_fraud_indicators(bursty))

        idx = FEATURE_NAMES.index("burst_concentration_ratio")
        self.assertGreater(v_bursty[idx], v_steady[idx])


class TestDecisionEngineIntegration(unittest.TestCase):
    def test_untrained_model_does_not_inflate_clean_candidates(self):
        """With no trained model the ML term is dropped and weights
        renormalised, so a clean candidate must still score Low."""
        from app.services.decision_engine import DecisionEngine

        fel = FeatureEngineeringLayer()
        gh = make_profile([3] * 104)
        fi = fel.compute_fraud_indicators(gh)
        tse = fel.compute_tse(gh, _NeutralQuality())
        nts = fel.compute_nts(gh)

        de = DecisionEngine()
        result = de.evaluate(tse, nts, fi, gh, resume=None, audio=None)

        self.assertEqual(result.fraud_risk_level, "Low")
        self.assertLessEqual(result.fraud_index, 1.0)


class _NeutralQuality:
    """Minimal stand-in for CodeQualityMetrics so these tests do not require
    a running SonarQube instance."""

    def quality_score(self) -> float:
        return 0.7


if __name__ == "__main__":
    unittest.main()
