"""
Claim consistency assessment for candidates whose GitHub evidence is NOT
available — either the account resolves but is verifiably empty (0 repos,
0 commits — a real, checkable NEGATIVE fact), or the handle doesn't resolve
at all (no fact available either way).

This is a distinct detection strategy from assess_claims() in
train_on_real_dataset.py, which compares claims against verified positive
GitHub evidence. Here there is no positive evidence to compare against, so
detection relies on:
  1. If GitHub is confirmed EMPTY: any claim of GitHub activity (repos,
     commits, stars, collaborators) directly contradicts a verified fact,
     exactly like evidence-based detection — the "evidence" is just a
     negative one.
  2. If GitHub is UNRESOLVED (can't confirm anything about it either way):
     GitHub-specific claims cannot be checked at all and must not be
     penalised. Detection instead relies on internal consistency — does the
     transcript contradict the résumé itself (experience length, project
     status, job title), and does the candidate's story hold together
     under follow-up rather than shifting or contradicting itself.

Reuses the same ClaimEvidenceAssessment output schema as
train_on_real_dataset.py so results are directly comparable and can feed
the same downstream feature/classifier code.
"""
import json
import time
from typing import Literal

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from app.services.llm_client import get_gemini_compatible_schema
from app.ml.train_on_real_dataset import ClaimEvidenceAssessment, GEMINI_MODEL, GEMINI_MAX_RETRIES

GithubStatus = Literal["empty", "unresolved"]

EMPTY_PROMPT = """You are a fraud-detection analyst for technical recruitment. You are given:
1. An interview TRANSCRIPT (what the candidate said, verbatim).
2. The candidate's RESUME text.
3. A VERIFIED FACT: GitHub was checked and the account resolves, but has ZERO
   repositories, ZERO commits, ZERO pull requests, and ZERO followers. This is a
   real, confirmed negative result — not missing data. The candidate cannot
   claim GitHub activity, repositories, stars, forks, or collaborators without
   directly contradicting this verified fact.

Compare the candidate's claims against this verified emptiness, and against
internal consistency with their own résumé. Look specifically for:
- Any claim of GitHub repositories, commits, stars, forks, or collaborators —
  these directly contradict the verified zero-activity account.
- Excuses for the discrepancy that are themselves implausible or shift under
  follow-up questioning (e.g. "must be private" when told private repos are
  counted; "GitHub reset my account"; contradicting an earlier claim).
- Experience, seniority, or role claims that contradict what the résumé itself
  states (this check applies regardless of GitHub).

Do NOT penalise the candidate for admitting the account is unused/empty, or
for honestly explaining that their real work lives elsewhere (a previous
employer's private repos, university group repos under a teammate's account)
— only score claims that assert something the verified zero-activity fact or
the résumé directly contradicts, and that are not walked back.

TRANSCRIPT:
{transcript}

RESUME:
{resume}

Return a structured assessment."""

UNRESOLVED_PROMPT = """You are a fraud-detection analyst for technical recruitment. You are given:
1. An interview TRANSCRIPT (what the candidate said, verbatim).
2. The candidate's RESUME text.
3. IMPORTANT: the GitHub handle on this candidate's résumé could NOT be resolved
   at all — no account was found. There is NO evidence, positive or negative,
   about this candidate's GitHub activity. You cannot confirm or deny any claim
   about their repositories, commits, or GitHub activity — do not penalise
   GitHub-related claims either way, since there is nothing to check them
   against.

Since GitHub cannot be verified, detection here relies entirely on INTERNAL
CONSISTENCY. Look specifically for:
- Claims in the transcript that directly contradict what the résumé itself
  states (years of experience, job title, employment dates, whether a project
  is complete/published vs. in-progress).
- A story that shifts or contradicts itself under follow-up questioning within
  the transcript (e.g. first claiming a profile is "public and starred", then
  claiming it must be "private" when pressed, without resolving the
  contradiction).
- Claims whose own internal logic doesn't hold together (e.g. calling
  in-progress research "effectively published").

Do NOT penalise the candidate for honestly admitting the GitHub account is
unused, unlinked, or was a mistake to list — that is not a contradiction, it
is a disclosure. Only score claims that remain internally inconsistent or
unresolved after follow-up.

TRANSCRIPT:
{transcript}

RESUME:
{resume}

Return a structured assessment."""


def assess_no_evidence(
    client: genai.Client, transcript: str, resume: str, github_status: GithubStatus
) -> ClaimEvidenceAssessment:
    prompt_template = EMPTY_PROMPT if github_status == "empty" else UNRESOLVED_PROMPT
    prompt = prompt_template.format(transcript=transcript, resume=resume)
    schema = get_gemini_compatible_schema(ClaimEvidenceAssessment)

    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json", response_schema=schema, temperature=0.1
                ),
            )
            return ClaimEvidenceAssessment.model_validate(json.loads(resp.text))
        except genai_errors.ClientError as e:
            is_rate_limit = getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e)
            if is_rate_limit and attempt < GEMINI_MAX_RETRIES - 1:
                wait = 20.0 * (attempt + 1)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Gemini rate-limit retries exhausted.")


def classify_github_status(evidence: dict) -> GithubStatus:
    """Given a parsed GitHub_Evidence dict, decides which no-evidence prompt applies."""
    if not evidence.get("profile_resolved", True):
        return "unresolved"
    return "empty"
