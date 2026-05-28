import re
from typing import Dict, Any, List, Optional, Tuple
from app.models.resume import ExtractedResumeSchema, TraceSpan, WorkExperienceItem, EducationItem

class ExplainabilityEngine:
    """Computes explainable AI traces and overall confidence metrics.
    
    Verifies extracted fields against original text and checks for semantic anomalies
    (e.g., date mismatches) to prepare for fraud detection.
    """

    @staticmethod
    def _find_snippet_in_text(value: str, full_text: str, context_window: int = 100) -> Optional[str]:
        """Finds the context snippet in the full text containing the given value."""
        if not value or len(value.strip()) < 2:
            return None
            
        value_clean = value.strip()
        
        # Exact substring match (case-insensitive)
        idx = full_text.lower().find(value_clean.lower())
        if idx != -1:
            start = max(0, idx - context_window)
            end = min(len(full_text), idx + len(value_clean) + context_window)
            snippet = full_text[start:end].replace("\n", " ").strip()
            return f"... {snippet} ..."
            
        # Try finding based on partial token matching (split by spaces)
        tokens = [t for t in value_clean.split() if len(t) > 2]
        if tokens:
            # Look for the occurrence of the first two tokens within proximity
            pattern = r"\b" + re.escape(tokens[0]) + r"\b"
            for match in re.finditer(pattern, full_text, re.IGNORECASE):
                start_idx = match.start()
                # Grab a wider block of text
                sub_start = max(0, start_idx - 50)
                sub_end = min(len(full_text), start_idx + 150)
                candidate_snippet = full_text[sub_start:sub_end]
                
                # Check if other tokens are in this candidate
                match_count = sum(1 for t in tokens[1:] if t.lower() in candidate_snippet.lower())
                if match_count >= len(tokens) // 2:
                    return f"... {candidate_snippet.replace(chr(10), ' ').strip()} ..."
                    
        return None

    def trace_entities(self, entities: ExtractedResumeSchema, full_text: str) -> List[TraceSpan]:
        """Maps extracted entities back to the source text to create proof traces."""
        traces: List[TraceSpan] = []

        # Trace root-level text fields
        root_fields = ["full_name", "email", "phone", "github", "linkedin", "claimed_role"]
        for field in root_fields:
            val = getattr(entities, field)
            if val:
                snippet = self._find_snippet_in_text(val, full_text)
                traces.append(TraceSpan(
                    field=field,
                    extracted_value=val,
                    raw_snippet=snippet or "Not traceably found in raw text (potential inference or format normalization)",
                    confidence=1.0 if snippet else 0.4
                ))

        # Trace education items
        for idx, edu in enumerate(entities.education):
            # Trace school
            snippet = self._find_snippet_in_text(edu.institution, full_text)
            traces.append(TraceSpan(
                field=f"education[{idx}].institution",
                extracted_value=edu.institution,
                raw_snippet=snippet or "Not traceably found",
                confidence=1.0 if snippet else 0.3
            ))

        # Trace work experience items
        for idx, work in enumerate(entities.work_experience):
            snippet = self._find_snippet_in_text(work.company, full_text)
            traces.append(TraceSpan(
                field=f"work_experience[{idx}].company",
                extracted_value=work.company,
                raw_snippet=snippet or "Not traceably found",
                confidence=1.0 if snippet else 0.3
            ))

        # Trace projects
        for idx, proj in enumerate(entities.projects):
            snippet = self._find_snippet_in_text(proj.name, full_text)
            traces.append(TraceSpan(
                field=f"projects[{idx}].name",
                extracted_value=proj.name,
                raw_snippet=snippet or "Not traceably found",
                confidence=1.0 if snippet else 0.3
            ))

        return traces

    @staticmethod
    def _parse_year(date_str: Optional[str]) -> Optional[int]:
        """Extracts a 4-digit year from a date string for logical validation."""
        if not date_str:
            return None
        match = re.search(r"\b(19\d{2}|20\d{2})\b", date_str)
        if match:
            return int(match.group(1))
        if "present" in date_str.lower() or "current" in date_str.lower():
            # Treat present as current year
            import datetime
            return datetime.datetime.now().year
        return None

    def evaluate_confidence(
        self, 
        entities: ExtractedResumeSchema, 
        anchors: Dict[str, Any], 
        traces: List[TraceSpan]
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculates a hybrid confidence score and raises validation warnings.
        
        This structure prepares metrics directly consumable by fraud detection engines.
        """
        validation_report: Dict[str, Any] = {
            "email_verified": False,
            "phone_verified": False,
            "github_verified": False,
            "linkedin_verified": False,
            "date_anomalies_detected": False,
            "anomalies": []
        }

        # 1. Contact Verification (Regex Anchors vs LLM Extractions)
        # Check email
        extracted_email = entities.email.strip().lower() if entities.email else ""
        if extracted_email and any(extracted_email in a.lower() for a in anchors.get("emails", [])):
            validation_report["email_verified"] = True
            
        # Check phone
        extracted_phone = re.sub(r"\D", "", entities.phone) if entities.phone else ""
        anchor_phones = [re.sub(r"\D", "", p) for p in anchors.get("phones", [])]
        if extracted_phone and any(extracted_phone in ap or ap in extracted_phone for ap in anchor_phones):
            validation_report["phone_verified"] = True
            
        # Check GitHub profile URL
        extracted_gh = entities.github.strip().lower() if entities.github else ""
        if extracted_gh:
            gh_username = extracted_gh.split("/")[-1] if "/" in extracted_gh else extracted_gh
            for gh_anchor in anchors.get("github_links", []):
                if gh_username in gh_anchor.lower():
                    validation_report["github_verified"] = True
                    break

        # Check LinkedIn profile URL
        extracted_li = entities.linkedin.strip().lower() if entities.linkedin else ""
        if extracted_li:
            li_username = extracted_li.split("/")[-1] if "/" in extracted_li else extracted_li
            for li_anchor in anchors.get("linkedin_links", []):
                if li_username in li_anchor.lower():
                    validation_report["linkedin_verified"] = True
                    break

        # 2. Chronological/Date Consistency Checks
        # Verify employment start/end dates do not conflict
        for idx, work in enumerate(entities.work_experience):
            start_yr = self._parse_year(work.start_date)
            end_yr = self._parse_year(work.end_date)
            if start_yr and end_yr and start_yr > end_yr:
                validation_report["date_anomalies_detected"] = True
                validation_report["anomalies"].append(
                    f"Employment chronological conflict at '{work.company}': Start year ({start_yr}) is after end year ({end_yr})."
                )

        # Verify education start/end dates do not conflict
        for idx, edu in enumerate(entities.education):
            start_yr = self._parse_year(edu.start_date)
            end_yr = self._parse_year(edu.end_date)
            if start_yr and end_yr and start_yr > end_yr:
                validation_report["date_anomalies_detected"] = True
                validation_report["anomalies"].append(
                    f"Education chronological conflict at '{edu.institution}': Start year ({start_yr}) is after end year ({end_yr})."
                )

        # 3. Compute Score
        # Trace confidence
        found_traces = sum(1 for t in traces if "Not traceably found" not in t.raw_snippet)
        trace_ratio = found_traces / len(traces) if traces else 0.0
        
        # Anchor verification ratio
        anchor_checks = [
            validation_report["email_verified"],
            validation_report["phone_verified"],
            validation_report["github_verified"] if entities.github else True,
            validation_report["linkedin_verified"] if entities.linkedin else True
        ]
        anchor_score = sum(1 for c in anchor_checks if c) / len(anchor_checks)
        
        # Anomaly penalties
        penalty = 0.0
        if validation_report["date_anomalies_detected"]:
            penalty += 0.15 * len(validation_report["anomalies"])
            
        # Combine metrics into a weighted confidence index
        # 60% trace lookup + 40% regex anchor alignment - penalties
        raw_score = (0.6 * trace_ratio) + (0.4 * anchor_score) - penalty
        confidence_score = max(0.0, min(1.0, raw_score))
        
        return round(confidence_score, 2), validation_report
