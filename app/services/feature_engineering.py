import logging
import re
import statistics
from typing import List, Dict, Any

from app.models.github_metrics import GitHubProfileMetrics
from app.models.evaluation import (
    TechnicalSkillExpertise,
    NonTechnicalSkills,
    FraudIndicators,
)
from app.services.sca_agent import CodeQualityMetrics

logger = logging.getLogger(__name__)


class FeatureEngineeringLayer:
    """Converts raw MSR and SCA signals into normalised feature vectors.

    Three groups of features are produced:
      1. TechnicalSkillExpertise (TSE)
      2. NonTechnicalSkills      (NTS)
      3. FraudIndicators         (FI)

    All scores are normalised to the 0.0 – 1.0 range so that the
    Decision Engine can combine them with consistent weights.
    """

    # ---------------------------------------------------------------
    # Thresholds used for normalisation / fraud detection
    # ---------------------------------------------------------------
    # Healthy commits-per-week for an active developer
    NORMAL_COMMIT_FREQUENCY = 10.0

    # If > this fraction of commits happen in a single burst window
    # we flag it as suspicious
    BURST_COMMIT_THRESHOLD = 0.40

    # If a commit has an unusually large size (lines changed) it may
    # indicate a bulk copy-paste
    LARGE_COMMIT_SIZE_THRESHOLD = 500

    # The bulk-dump pattern this rule targets is "a handful of enormous
    # commits, no sustained history" (see synthetic_data.py's bulk_dump
    # archetype: 2-5 commits total: and the test fixture: 3 commits total).
    # Validated against 8 real GitHub accounts: several legitimate
    # developers with 41-289 total commits and ordinary small-project
    # activity (large occasional commits, e.g. an initial project-skeleton
    # push) were tripping this rule on average-size + low-frequency alone.
    # Gating on a low total-commit count is what actually distinguishes
    # "a few oversized commits and nothing else" from "a real, if modest,
    # commit history that happens to include some large commits."
    BULK_DUMP_MAX_TOTAL_COMMITS = 20

    # Minimum meaningful PR approval rate
    MIN_ACCEPTABLE_APPROVAL_RATE = 0.50

    # ---------------------------------------------------------------
    # 1.  Technical Skill Expertise
    # ---------------------------------------------------------------

    def compute_tse(
        self,
        gh: GitHubProfileMetrics,
        code_quality: CodeQualityMetrics,
    ) -> TechnicalSkillExpertise:
        """Compute the Technical Skill Expertise (TSE) score.

        Sub-dimensions:
          language_proficiency  – breadth and depth of languages used
          pr_quality            – PR approval rate and comment density
          code_quality          – SonarQube composite quality score
          efficiency            – commit frequency vs. commit size ratio
        """
        # ---- Language Proficiency --------------------------------
        # Score based on number of distinct languages used
        num_languages = len(gh.repository_metadata.primary_languages)
        # Normalise: 1 lang = 0.1, 5+ langs = 1.0
        language_proficiency = min(num_languages / 5.0, 1.0)

        # ---- PR Quality -----------------------------------------
        pr = gh.pull_requests
        if pr.total_prs == 0:
            pr_quality = 0.5  # Neutral — no PRs to evaluate
        else:
            # Higher approval rate = better quality
            # Penalise low comment density (< 1 comment/PR = superficial review)
            comment_bonus = min(pr.average_comments_per_pr / 3.0, 0.2)
            pr_quality = min(pr.approval_rate + comment_bonus, 1.0)

        # ---- Code Quality (from SonarQube) ----------------------
        cq_score = code_quality.quality_score()

        # ---- Efficiency -----------------------------------------
        ch = gh.commit_history
        if ch.total_commits == 0:
            efficiency = 0.0
        else:
            # Penalise very large average commit sizes (bulk imports)
            size_penalty = min(
                ch.average_commit_size_lines / self.LARGE_COMMIT_SIZE_THRESHOLD, 1.0
            )
            # Reward steady commit frequency up to the normal threshold
            freq_score = min(
                ch.frequency_per_week / self.NORMAL_COMMIT_FREQUENCY, 1.0
            )
            efficiency = freq_score * (1.0 - 0.3 * size_penalty)

        overall = round(
            0.25 * language_proficiency
            + 0.30 * pr_quality
            + 0.30 * cq_score
            + 0.15 * efficiency,
            3,
        )

        return TechnicalSkillExpertise(
            language_proficiency_score=round(language_proficiency, 3),
            pr_quality_score=round(pr_quality, 3),
            code_quality_score=round(cq_score, 3),
            efficiency_score=round(efficiency, 3),
            overall_tse_score=overall,
        )

    # ---------------------------------------------------------------
    # 2.  Non-Technical Skills
    # ---------------------------------------------------------------

    def compute_nts(self, gh: GitHubProfileMetrics) -> NonTechnicalSkills:
        """Compute Non-Technical Skills (NTS) score.

        Sub-dimensions:
          collaboration     – PR review engagement (comments / PRs)
          req_understanding – ratio of accepted PRs (shows ability to
                              understand specs and deliver correctly)
          project_mgmt      – number of owned repos as a proxy
          motivation        – language diversity + contributed projects
        """
        pr = gh.pull_requests
        repo = gh.repository_metadata

        # ---- Collaboration -----------------------------------------------
        if pr.total_prs == 0:
            collaboration = 0.3  # Neutral
        else:
            # High comment density suggests active code review participation
            collaboration = min(pr.average_comments_per_pr / 5.0, 1.0)

        # ---- Requirements Understanding ---------------------------------
        req_understanding = pr.approval_rate  # Already 0-1

        # ---- Project Management ----------------------------------------
        # Proxy: number of repos owned (capped at 20 for normalisation)
        project_mgmt = min(repo.total_repos / 20.0, 1.0)

        # ---- Motivation / Hustle ----------------------------------------
        # Breadth: how many distinct languages they've programmed in
        lang_breadth = min(len(repo.primary_languages) / 6.0, 1.0)
        # Popularity signals effort / visibility
        social_score = min((repo.total_stars + repo.total_forks) / 100.0, 1.0)
        motivation = round(0.6 * lang_breadth + 0.4 * social_score, 3)

        overall = round(
            0.30 * collaboration
            + 0.30 * req_understanding
            + 0.20 * project_mgmt
            + 0.20 * motivation,
            3,
        )

        return NonTechnicalSkills(
            collaboration_proficiency=round(collaboration, 3),
            requirements_understanding=round(req_understanding, 3),
            project_management_skills=round(project_mgmt, 3),
            motivation_hustle=round(motivation, 3),
            overall_nts_score=overall,
        )

    # ---------------------------------------------------------------
    # 3.  Fraud Indicators
    # ---------------------------------------------------------------

    # Minimum number of active weeks needed before we trust the
    # concentration/variation statistics (too few points => noisy).
    MIN_ACTIVE_WEEKS_FOR_BURST_STATS = 3
    BURST_CONCENTRATION_THRESHOLD = 0.55  # top 10% of weeks holding >55% of commits

    def compute_fraud_indicators(
        self, gh: GitHubProfileMetrics
    ) -> FraudIndicators:
        """Detect suspicious activity patterns from MSR data.

        Checks performed:
          Burst Commits   – A large fraction of all commits arriving in a
                            small handful of weeks (profile padding right
                            before a job hunt), measured from the *real*
                            weekly commit time series rather than an average.
          Fake Activity   – Extreme commit size with very low sustained
                            frequency (single massive push with no organic
                            history).
          Message Quality – Generic / auto-generated commit messages.
          PR Rejection    – Unusually high rejection rate on submitted PRs.

        All continuous sub-scores are also stored on `FraudIndicators` so the
        ML-based anomaly detector (see anomaly_model.py) can reuse the exact
        same evidence instead of recomputing it differently.
        """
        flags: List[str] = []
        anomaly_score = 0.0
        ch = gh.commit_history
        weeks = ch.weekly_commit_counts

        # ---- Burst Commit Detection (real weekly time series) ---------
        burst_detected = False
        burst_concentration_ratio = 0.0
        commit_cv = 0.0

        active_weeks = [c for c in weeks if c > 0]
        # The sample-size guard is on the length of the observed calendar
        # window, NOT on the number of active weeks. Guarding on active weeks
        # is self-defeating: the more concentrated the burst, the fewer active
        # weeks it has, so the most extreme padding pattern (every commit in a
        # single week) would be the one case that never gets measured.
        if len(weeks) >= self.MIN_ACTIVE_WEEKS_FOR_BURST_STATS and ch.total_commits > 0:
            # What share of all commits landed in the busiest 10% of active weeks?
            top_n = max(1, round(len(active_weeks) * 0.10))
            top_weeks_sum = sum(sorted(active_weeks, reverse=True)[:top_n])
            burst_concentration_ratio = round(top_weeks_sum / ch.total_commits, 3)

            # Coefficient of variation across the FULL calendar series
            # (including zero weeks) — a genuinely active contributor has
            # a low CV (steady drip of commits); someone padding their
            # profile right before applying has a high CV (long silence,
            # then a spike).
            mean_w = statistics.mean(weeks)
            if mean_w > 0:
                commit_cv = round(statistics.pstdev(weeks) / mean_w, 3)

            if (
                burst_concentration_ratio >= self.BURST_CONCENTRATION_THRESHOLD
                and ch.total_commits >= 20
            ):
                burst_detected = True
                flags.append(
                    f"Burst commit pattern detected: {burst_concentration_ratio:.0%} of "
                    f"{ch.total_commits} commits were made in just the busiest "
                    f"{top_n} week(s) out of {len(active_weeks)} active weeks."
                )
                # Scale contribution smoothly instead of a flat add, so a
                # borderline case (just over threshold) isn't penalised as
                # heavily as an extreme one (e.g. 95% concentration).
                anomaly_score += 0.35 * min(
                    (burst_concentration_ratio - self.BURST_CONCENTRATION_THRESHOLD)
                    / (1.0 - self.BURST_CONCENTRATION_THRESHOLD),
                    1.0,
                )

        # ---- Fake / Bulk Activity Detection --------------------------
        # True calendar average (commits / total elapsed weeks, including
        # silent weeks) — a more honest "sustained effort" signal than the
        # active-week-only average used elsewhere for UI display.
        true_calendar_freq = (
            ch.total_commits / len(weeks) if weeks else ch.frequency_per_week
        )
        fake_activity = False
        if (
            ch.average_commit_size_lines > self.LARGE_COMMIT_SIZE_THRESHOLD
            and true_calendar_freq < 1.0
            and ch.total_commits < self.BULK_DUMP_MAX_TOTAL_COMMITS
        ):
            fake_activity = True
            flags.append(
                f"Suspicious bulk commit activity: avg size "
                f"{ch.average_commit_size_lines:.0f} lines/commit "
                f"with only {true_calendar_freq:.2f} commits/week sustained "
                "over the observed history."
            )
            anomaly_score += 0.30

        # ---- Auto-generated / Low-quality Commit Messages ------------
        generic_ratio = 0.0
        if ch.commit_messages:
            generic_patterns = [
                r"^initial commit$",
                r"^update$",
                r"^fix$",
                r"^test$",
                r"^wip$",
                r"^commit\s*\d*$",
                r"^\.+$",
            ]
            generic_count = sum(
                1
                for msg in ch.commit_messages
                if any(
                    re.match(p, msg.strip().lower())
                    for p in generic_patterns
                )
            )
            generic_ratio = round(generic_count / len(ch.commit_messages), 3)
            if generic_ratio > 0.60:
                flags.append(
                    f"Low-quality commit messages: {generic_ratio:.0%} of sampled "
                    "messages are generic/auto-generated."
                )
                anomaly_score += 0.20

        # ---- PR Rejection Rate Anomaly ------------------------------
        pr = gh.pull_requests
        pr_rejection_rate = 0.0
        if pr.total_prs > 5:
            pr_rejection_rate = round(1.0 - pr.approval_rate, 3)
            if pr.approval_rate < self.MIN_ACCEPTABLE_APPROVAL_RATE:
                flags.append(
                    f"High PR rejection rate: only {pr.approval_rate:.0%} approval "
                    f"across {pr.total_prs} pull requests."
                )
                anomaly_score += 0.15

        final_score = round(min(anomaly_score, 1.0), 3)
        logger.info(
            f"Fraud Indicators: score={final_score}, burst={burst_detected} "
            f"(concentration={burst_concentration_ratio}, cv={commit_cv}), "
            f"fake_activity={fake_activity}, flags={len(flags)}"
        )

        return FraudIndicators(
            burst_commits_detected=burst_detected,
            fake_activity_detected=fake_activity,
            anomaly_score=final_score,
            flags=flags,
            burst_concentration_ratio=burst_concentration_ratio,
            commit_coefficient_variation=commit_cv,
            generic_message_ratio=generic_ratio,
            pr_rejection_rate=pr_rejection_rate,
        )
