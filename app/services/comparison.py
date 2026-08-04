import logging
import re
from typing import Dict, Any, List, Optional
from app.models.ccs import CCSJobDocument
from app.models.resume import ResumeDocument

logger = logging.getLogger(__name__)

class ClaimComparisonService:
    """Performs semantic cross-validation and mismatch detection between verbal interview claims, resume facts, and GitHub evidence."""

    def __init__(self):
        pass

    def compare_interview_vs_resume(
        self, 
        ccs_job: CCSJobDocument, 
        resume: ResumeDocument
    ) -> Dict[str, Any]:
        """Compares interview CCS claims against parsed resume entities.
        
        Detects skill gaps, experience mismatches, and structural contradictions.
        """
        if not ccs_job.claims:
            return {"status": "error", "message": "No claims available to compare."}

        interview_claims = ccs_job.claims
        resume_entities = resume.entities

        # 1. Compare technology sets
        interview_tech = set(
            interview_claims.technical_skills +
            interview_claims.frameworks +
            interview_claims.tools +
            interview_claims.programming_languages +
            interview_claims.cloud_platforms +
            interview_claims.databases
        )
        interview_tech = {t.lower() for t in interview_tech}

        resume_tech = {s.lower() for s in resume_entities.skills}
        for exp in resume_entities.work_experience:
            for t in exp.technologies:
                resume_tech.add(t.lower())
        for proj in resume_entities.projects:
            for t in proj.technologies:
                resume_tech.add(t.lower())

        # Unmentioned Resume Skills (inflation check)
        unmentioned_skills = list(resume_tech - interview_tech)
        
        # New verbal skills mentioned (enrichment or exaggeration)
        new_verbal_skills = list(interview_tech - resume_tech)

        # 2. Years of Experience Check
        experience_discrepancy = False
        discrepancy_details = ""
        
        # Parse years of experience from interview
        interview_years = self._parse_years(interview_claims.years_of_experience)
        resume_years = self._parse_years(resume_entities.total_years_experience)

        if interview_years is not None and resume_years is not None:
            diff = abs(interview_years - resume_years)
            if diff >= 2.0:
                experience_discrepancy = True
                discrepancy_details = (
                    f"Candidate verbally claimed '{interview_claims.years_of_experience}' "
                    f"({interview_years} years) of experience, but the resume states "
                    f"'{resume_entities.total_years_experience}' ({resume_years} years)."
                )

        # 3. Contradiction Detection & Explainable AI output
        contradictions = []
        if experience_discrepancy:
            contradictions.append(discrepancy_details)
            
        # Check if candidate claims leadership but resume lists no leadership roles
        interview_leadership = len(interview_claims.leadership_claims) > 0
        resume_leadership = False
        leadership_keywords = ["lead", "manager", "director", "head", "architect", "managed", "led"]
        
        for exp in resume_entities.work_experience:
            role_lower = exp.role.lower()
            if any(k in role_lower for k in leadership_keywords):
                resume_leadership = True
                break
                
        if interview_leadership and not resume_leadership:
            contradictions.append(
                "Candidate verbally claimed leadership experience (e.g. "
                f"'{interview_claims.leadership_claims[0]}'), but no leadership titles "
                "or roles were detected in their resume work history."
            )

        # Calculate a mismatch index (0.0 to 1.0, where 1.0 means high conflict)
        mismatch_index = 0.0
        if len(interview_tech) > 0:
            # Fraction of interview technologies NOT on resume
            mismatch_index += 0.4 * (len(new_verbal_skills) / max(len(interview_tech), 1))
        if experience_discrepancy:
            mismatch_index += 0.4
        if interview_leadership and not resume_leadership:
            mismatch_index += 0.2

        mismatch_index = min(1.0, round(mismatch_index, 2))

        return {
            "comparison_type": "interview_vs_resume",
            "mismatch_index": mismatch_index,
            "common_skills": list(interview_tech.intersection(resume_tech)),
            "unmentioned_resume_skills": unmentioned_skills,
            "new_verbal_skills": new_verbal_skills,
            "experience_discrepancy": experience_discrepancy,
            "contradictions": contradictions,
            "summary": (
                f"Comparison finished. Mismatch index is {mismatch_index}. "
                f"Found {len(contradictions)} contradictions, "
                f"{len(unmentioned_skills)} unmentioned resume skills, and "
                f"{len(new_verbal_skills)} new verbal skills."
            )
        }

    def compare_interview_vs_github(
        self, 
        ccs_job: CCSJobDocument, 
        github_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compares interview CCS claims against retrieved GitHub profile evidence.
        
        Args:
            ccs_job: The parsed CCS document.
            github_profile: Dictionary containing:
                - repos: List[Dict] with 'name', 'language', 'stargazers_count'
                - languages: Dict[str, int] representing bytes of code per language
                - total_commits: int
                - total_prs: int
        """
        if not ccs_job.claims:
            return {"status": "error", "message": "No claims available to compare."}

        interview_claims = ccs_job.claims
        
        # Normalize GitHub languages
        github_languages = {k.lower(): v for k, v in github_profile.get("languages", {}).items()}
        github_repo_names = [r.get("name", "").lower() for r in github_profile.get("repos", [])]
        
        contradictions = []
        missing_evidence = []
        
        # Check claimed languages vs GitHub codebases
        claimed_languages = [lang.lower() for lang in interview_claims.programming_languages]
        for lang in claimed_languages:
            if lang not in github_languages:
                missing_evidence.append(
                    f"Candidate verbally claimed '{lang}' proficiency, but no matching code "
                    "or repositories were found in their GitHub account."
                )

        # Check cloud & infrastructure claims (e.g. Docker, Kubernetes, AWS) vs project files or topics
        infra_claims = [c.lower() for c in (interview_claims.cloud_platforms + interview_claims.tools)]
        devops_keywords = ["docker", "kubernetes", "k8s", "aws", "terraform", "ansible", "jenkins", "ci", "cd"]
        
        has_devops_claim = any(k in infra_claims for k in devops_keywords)
        has_devops_repo = any(any(k in name for k in devops_keywords) for name in github_repo_names)
        
        if has_devops_claim and not has_devops_repo:
            contradictions.append(
                "Candidate verbally claimed DevOps / Cloud experience (e.g. Docker, Kubernetes, AWS), "
                "but no devops-themed repositories, Dockerfiles, or infra configuration scripts "
                "were identified in their public GitHub profile."
            )

        # Compute conflict/mismatch score (0.0 to 1.0)
        github_conflict_score = 0.0
        if claimed_languages:
            missing_count = sum(1 for lang in claimed_languages if lang not in github_languages)
            github_conflict_score += 0.5 * (missing_count / len(claimed_languages))
            
        if has_devops_claim and not has_devops_repo:
            github_conflict_score += 0.3
            
        # Check overall activity
        total_commits = github_profile.get("total_commits", 0)
        if total_commits < 10 and len(claimed_languages) > 1:
            github_conflict_score += 0.2
            contradictions.append(
                f"Candidate claimed extensive technical skills, but their public GitHub "
                f"activity is extremely low ({total_commits} total commits)."
            )

        github_conflict_score = min(1.0, round(github_conflict_score, 2))

        return {
            "comparison_type": "interview_vs_github",
            "conflict_score": github_conflict_score,
            "missing_evidence": missing_evidence,
            "contradictions": contradictions,
            "summary": (
                f"GitHub validation complete. Conflict score: {github_conflict_score}. "
                f"Detected {len(contradictions)} structural conflicts and {len(missing_evidence)} "
                "cases of missing technical evidence."
            )
        }

    def _parse_years(self, duration_str: Optional[str]) -> Optional[float]:
        """Helper to parse a duration string into a float representing years."""
        if not duration_str:
            return None
            
        duration_str = duration_str.lower().strip()
        
        # Regex to find numbers
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", duration_str)
        if not numbers:
            return None
            
        val = float(numbers[0])
        
        if "month" in duration_str:
            return val / 12.0
        return val
