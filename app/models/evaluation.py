from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class TechnicalSkillExpertise(BaseModel):
    language_proficiency_score: float = 0.0
    pr_quality_score: float = 0.0
    code_quality_score: float = 0.0
    efficiency_score: float = 0.0
    overall_tse_score: float = 0.0

class NonTechnicalSkills(BaseModel):
    collaboration_proficiency: float = 0.0
    requirements_understanding: float = 0.0
    project_management_skills: float = 0.0
    motivation_hustle: float = 0.0
    overall_nts_score: float = 0.0

class FraudIndicators(BaseModel):
    burst_commits_detected: bool = False
    fake_activity_detected: bool = False
    anomaly_score: float = 0.0
    flags: List[str] = Field(default_factory=list)

    # Raw, continuous signals (0.0-1.0 unless noted) kept alongside the
    # boolean flags above. These are consumed directly as features by the
    # ML-based anomaly detector (IsolationForest) in anomaly_model.py, so
    # the rule-based layer and the ML layer are looking at the same
    # underlying evidence rather than duplicating logic.
    burst_concentration_ratio: float = 0.0  # share of commits packed into the busiest 10% of active weeks
    commit_coefficient_variation: float = 0.0  # std/mean of the weekly commit series (bursty vs steady)
    generic_message_ratio: float = 0.0  # fraction of sampled commit messages that are generic/auto-generated
    pr_rejection_rate: float = 0.0  # 1 - approval_rate, only meaningful when total_prs > 0

class DecisionEngineResult(BaseModel):
    technical_skill_score: float = 0.0
    collaboration_consistency_score: float = 0.0
    fraud_risk_level: str = "Low"  # "Low", "Medium", "High"
    fraud_index: float = 0.0  # continuous 0.0-1.0 score behind fraud_risk_level
    ml_anomaly_score: float = 0.0  # IsolationForest component of fraud_index, for transparency
    confidence_score: float = 0.0
    summary_rationale: str = ""
