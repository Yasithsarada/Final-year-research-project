import logging
from typing import List, Set, Optional

from app.models.evaluation import (
    TechnicalSkillExpertise,
    NonTechnicalSkills,
    FraudIndicators,
    DecisionEngineResult,
)
from app.models.resume import ResumeDocument, NormalizedSkill
from app.models.ccs import CCSJobDocument, CCSClaimSet
from app.models.github_metrics import GitHubProfileMetrics
from app.services.anomaly_model import MLAnomalyDetector, build_feature_vector

logger = logging.getLogger(__name__)


class DecisionEngine:
    """Fraud & Anomaly Detection Model + Decision Engine.

    Combines all engineered feature vectors with the candidate's
    self-reported claims (from CV and Audio Interview) to produce
    three final outputs:

      1. Technical Skill Score      (TSS)  – 0.0 to 1.0
      2. Collaboration & Consistency Score (CCS) – 0.0 to 1.0
      3. Fraud Risk Level           (FRL)  – "Low" | "Medium" | "High"

    Fraud Risk is deep: it cross-checks:
      - Claimed skills vs. GitHub evidence (language usage)
      - Claimed years of experience vs. commit history timeline
      - Audio interview claims vs. resume entities
      - Raw Fraud Indicators from the FEL (burst commits, etc.)
    """

    # ---------------------------------------------------------------
    # Thresholds
    # ---------------------------------------------------------------
    FRAUD_HIGH_THRESHOLD = 0.60
    FRAUD_MEDIUM_THRESHOLD = 0.30

    # Penalty applied per skill claimed but not evidenced in GitHub
    UNVERIFIED_SKILL_PENALTY = 0.05

    def __init__(self):
        # Loads the trained IsolationForest once per DecisionEngine instance.
        # If no model has been trained yet, MLAnomalyDetector falls back to a
        # neutral 0.5 score so the pipeline still runs end-to-end (see
        # app/ml/train_anomaly_model.py to train one).
        self.ml_detector = MLAnomalyDetector()

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _normalise_skill_set(skills: List[str]) -> Set[str]:
        """Lowercase and strip skills for fuzzy matching."""
        cleaned = set()
        for s in skills:
            cleaned.add(s.strip().lower().replace("-", "").replace(".", "").replace(" ", ""))
        return cleaned

    @staticmethod
    def _skill_overlap_ratio(claimed: Set[str], evidenced: Set[str]) -> float:
        """What fraction of claimed skills are backed by evidence?"""
        if not claimed:
            return 1.0  # Nothing claimed → no mismatch
        matched = claimed & evidenced
        return len(matched) / len(claimed)

    # ---------------------------------------------------------------
    # Skill Cross-Verification
    # ---------------------------------------------------------------

    def _verify_skills_against_github(
        self,
        claimed_skills: Set[str],
        gh: GitHubProfileMetrics,
    ) -> float:
        """Check whether claimed tech skills appear in GitHub language data.

        Returns a mismatch penalty score (0.0 = full match, 1.0 = total mismatch).
        """
        github_langs = self._normalise_skill_set(
            list(gh.repository_metadata.primary_languages.keys())
        )
        overlap = self._skill_overlap_ratio(claimed_skills, github_langs)
        mismatch = 1.0 - overlap
        if mismatch > 0.3:
            logger.info(
                f"Skill mismatch: {mismatch:.0%} of claimed skills "
                "not evidenced in GitHub repositories."
            )
        return round(mismatch, 3)

    def _verify_audio_vs_resume(
        self,
        resume: Optional[ResumeDocument],
        audio: Optional[CCSJobDocument],
    ) -> float:
        """Cross-check audio interview claims vs. resume entities.

        Returns a mismatch penalty (0.0 = consistent, 1.0 = contradictory).
        """
        if not resume or not audio or not audio.claims:
            return 0.0  # No data → assume consistent

        # Compare skills
        resume_skills = self._normalise_skill_set(
            [s.canonical for s in resume.normalized_skills]
            + resume.entities.skills
        )
        audio_skills = self._normalise_skill_set(
            audio.claims.technical_skills
            + audio.claims.programming_languages
            + audio.claims.frameworks
        )

        overlap = self._skill_overlap_ratio(audio_skills, resume_skills)
        mismatch = 1.0 - overlap

        if mismatch > 0.4:
            logger.info(
                f"Audio vs Resume mismatch: {mismatch:.0%} of interview "
                "claims not found in resume entities."
            )
        return round(mismatch, 3)

    # ---------------------------------------------------------------
    # Main Scoring
    # ---------------------------------------------------------------

    def evaluate(
        self,
        tse: TechnicalSkillExpertise,
        nts: NonTechnicalSkills,
        fi: FraudIndicators,
        gh: GitHubProfileMetrics,
        resume: Optional[ResumeDocument] = None,
        audio: Optional[CCSJobDocument] = None,
    ) -> DecisionEngineResult:
        """Run the full Decision Engine evaluation.

        Args:
            tse:    Technical Skill Expertise features from FEL.
            nts:    Non-Technical Skills features from FEL.
            fi:     Fraud Indicators from FEL.
            gh:     Raw GitHub metrics for cross-verification.
            resume: Optional parsed resume document.
            audio:  Optional audio interview CCS document.

        Returns:
            DecisionEngineResult with TSS, CCS, and FRL.
        """
        rationale_parts = []

        # -------------------------------------------------------
        # 1. Technical Skill Score (TSS)
        # -------------------------------------------------------
        # Base = TSE overall, adjusted by skill verification
        claimed_skills: Set[str] = set()
        if resume:
            claimed_skills |= self._normalise_skill_set(
                [s.canonical for s in resume.normalized_skills]
            )
        if audio and audio.claims:
            claimed_skills |= self._normalise_skill_set(
                audio.claims.technical_skills + audio.claims.programming_languages
            )

        skill_mismatch = self._verify_skills_against_github(claimed_skills, gh)
        skill_penalty = skill_mismatch * self.UNVERIFIED_SKILL_PENALTY * len(claimed_skills)
        tss = round(max(0.0, min(tse.overall_tse_score - skill_penalty, 1.0)), 3)

        if skill_mismatch > 0.3:
            rationale_parts.append(
                f"Skill mismatch penalty applied: {skill_mismatch:.0%} of "
                "claimed skills not evidenced in GitHub."
            )

        # -------------------------------------------------------
        # 2. Collaboration & Consistency Score (CCS)
        # -------------------------------------------------------
        audio_resume_mismatch = self._verify_audio_vs_resume(resume, audio)
        consistency_penalty = audio_resume_mismatch * 0.20

        ccs = round(max(0.0, min(nts.overall_nts_score - consistency_penalty, 1.0)), 3)

        if audio_resume_mismatch > 0.3:
            rationale_parts.append(
                f"Audio-Resume consistency penalty: {audio_resume_mismatch:.0%} "
                "mismatch between interview claims and CV data."
            )

        # -------------------------------------------------------
        # 3. Fraud Risk Level (FRL) — Deep Assessment
        # -------------------------------------------------------
        # Hybrid scoring: deterministic rule-based flags (fi.anomaly_score)
        # are blended with a genuine unsupervised ML anomaly score from the
        # IsolationForest (ml_anomaly_score), on top of the two
        # cross-verification mismatches. The rule-based component keeps the
        # result explainable (each flag has a plain-English reason); the ML
        # component catches combinations of subtly-off behaviour that no
        # single hand-written rule was designed to catch.
        feature_vector = build_feature_vector(gh, fi, skill_mismatch, audio_resume_mismatch)
        ml_anomaly_score = self.ml_detector.score(feature_vector)

        if self.ml_detector.is_fitted:
            raw_fraud_index = (
                0.30 * fi.anomaly_score         # Hard, explainable behavioural rules
                + 0.30 * ml_anomaly_score        # Unsupervised ML anomaly signal
                + 0.20 * skill_mismatch          # Skill exaggeration
                + 0.20 * audio_resume_mismatch   # Story inconsistency
            )
            rationale_parts.append(f"ML anomaly score (IsolationForest): {ml_anomaly_score:.3f}")
        else:
            # No trained model available. Rather than feeding in a neutral 0.5 —
            # which would silently add a constant ~0.15 to EVERY candidate's
            # fraud index and drag clean candidates toward the Medium band —
            # we drop the ML term and renormalise the remaining weights so they
            # still sum to 1.0. The index stays on the same 0-1 scale and the
            # thresholds below remain meaningful.
            raw_fraud_index = (
                0.30 * fi.anomaly_score
                + 0.20 * skill_mismatch
                + 0.20 * audio_resume_mismatch
            ) / 0.70
            rationale_parts.append(
                "NOTE: ML anomaly detector is untrained — fraud index computed from "
                "rule-based signals only (weights renormalised). Run "
                "`python -m app.ml.train_anomaly_model` to enable the ML component."
            )

        fraud_index = round(min(raw_fraud_index, 1.0), 3)

        if fraud_index >= self.FRAUD_HIGH_THRESHOLD:
            fraud_risk = "High"
            rationale_parts.append(
                f"HIGH FRAUD RISK (index={fraud_index}): Multiple strong "
                "anomaly signals detected across behavioural, skill, and "
                "interview consistency dimensions."
            )
        elif fraud_index >= self.FRAUD_MEDIUM_THRESHOLD:
            fraud_risk = "Medium"
            rationale_parts.append(
                f"MEDIUM FRAUD RISK (index={fraud_index}): Some anomaly "
                "signals detected; further manual review is recommended."
            )
        else:
            fraud_risk = "Low"
            rationale_parts.append(
                f"LOW FRAUD RISK (index={fraud_index}): Candidate profile "
                "is broadly consistent across all data sources."
            )

        # Append FEL flags for transparency
        for flag in fi.flags:
            rationale_parts.append(f"  ⚑ {flag}")

        # -------------------------------------------------------
        # 4. Overall Confidence in the Decision
        # -------------------------------------------------------
        # If we have both resume AND audio, confidence is higher
        data_completeness = sum([
            0.40,                            # GitHub data always present
            0.35 if resume else 0.0,
            0.25 if audio and audio.claims else 0.0,
        ])
        confidence = round(data_completeness, 2)

        logger.info(
            f"Decision Engine: TSS={tss}, CCS={ccs}, "
            f"FRL={fraud_risk} (index={fraud_index}), confidence={confidence}"
        )

        return DecisionEngineResult(
            technical_skill_score=tss,
            collaboration_consistency_score=ccs,
            fraud_risk_level=fraud_risk,
            fraud_index=fraud_index,
            ml_anomaly_score=round(ml_anomaly_score, 3),
            confidence_score=confidence,
            summary_rationale=" | ".join(rationale_parts),
        )
