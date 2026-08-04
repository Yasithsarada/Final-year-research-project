import logging
from typing import List, Dict, Any
from app.models.ccs import CCSClaimSet

logger = logging.getLogger(__name__)

class CCSExplainer:
    """Generates explainability reports linking claims back to the original transcript."""

    def __init__(self):
        pass

    def generate_explanation_report(self, claims: CCSClaimSet) -> List[str]:
        """Generates structured, human-readable explanations of the extracted claims.
        
        Highlights supporting evidence and confidence scores.
        """
        explanations = []

        # 1. Experience & Job Roles
        if claims.job_roles:
            roles_str = ", ".join(claims.job_roles)
            explanations.append(
                f"Candidate claimed the following job role(s): {roles_str}."
            )
        if claims.years_of_experience:
            explanations.append(
                f"Candidate verbally claimed to have: '{claims.years_of_experience}' of professional experience."
            )

        # 2. Main technical skills
        all_skills = (
            claims.programming_languages + 
            claims.frameworks + 
            claims.databases + 
            claims.cloud_platforms + 
            claims.tools + 
            claims.technical_skills
        )
        if all_skills:
            unique_skills = sorted(list(set(all_skills)))
            explanations.append(
                f"Candidate asserted proficiency in {len(unique_skills)} technologies, including: {', '.join(unique_skills[:10])}."
            )

        # 3. Leadership Claims
        if claims.leadership_claims:
            for claim in claims.leadership_claims:
                explanations.append(f"Leadership/Management Claim: '{claim}'.")

        # 4. Projects Claimed
        if claims.projects_claimed:
            for proj in claims.projects_claimed:
                explanations.append(f"Candidate claimed involvement in project: '{proj}'.")

        # 5. Connect Evidence and Confidence
        explanations.append(
            f"Overall self-reported claim extraction confidence score: {claims.confidence_score:.2f}."
        )
        
        if claims.supporting_sentences:
            explanations.append("Supporting sentences extracted as evidence:")
            for idx, sentence in enumerate(claims.supporting_sentences[:5], 1):
                explanations.append(f"  [{idx}] \"{sentence}\"")
                
        return explanations
