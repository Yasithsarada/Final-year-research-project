"""
Validates the no-evidence claim-consistency assessment (app/ml/no_evidence_
assessment.py) against the real people in datasets/interview_cv_github_fraud_
dataset.csv whose GitHub evidence is unusable for the normal pipeline:
  - Malith Weerarathne: account resolves but is verifiably EMPTY (0 repos,
    0 commits) — a real negative fact.
  - Theekshana Kulathunga: the GitHub handle does not resolve at ALL —
    no fact available either way.

Run from the `reasearch v2` project root:
    python -m app.ml.evaluate_no_evidence_model

READ THIS BEFORE QUOTING NUMBERS: there are only 6 real rows here (3 people
x... actually 2 people x 3 cases). This is a proof-of-concept sanity check
that the no-evidence detection mechanism produces directionally correct
scores on known real cases — it is NOT a validated classifier and n=6 from
2 real people cannot support one. Treat any accuracy figure printed here as
illustrative only.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from google import genai

from app.core.config import settings
import app.ml.pipeline_config as cfg
from app.ml.train_on_real_dataset import parse_github_evidence, ClaimEvidenceAssessment
from app.ml.no_evidence_assessment import assess_no_evidence, classify_github_status

DATASET_PATH = cfg.LABELED_DATASET_PATH
RESULTS_DIR = cfg.RESULTS_DIR
CACHE_PATH = RESULTS_DIR / "no_evidence_claims_cache.json"
OUTPUT_CSV = RESULTS_DIR / "no_evidence_results.csv"


def is_no_evidence_row(evidence: dict) -> bool:
    if not evidence.get("profile_resolved", True):
        return True
    return evidence.get("total_commits", 0) == 0 and evidence.get("total_repos", 0) == 0


def load_cache() -> dict:
    return json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def row_hash(row) -> str:
    import hashlib
    payload = f"{row['Transcript']}|{row['Resume']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=cfg.LABELED_DATASET_PATH,
                         help=f"Labeled dataset to scan for no-evidence rows (default {cfg.LABELED_DATASET_PATH}).")
    parser.add_argument("--results-dir", type=Path, default=cfg.RESULTS_DIR,
                         help=f"Where to write results/cache (default {cfg.RESULTS_DIR}).")
    parser.add_argument("--threshold", type=float, default=cfg.NO_EVIDENCE_RISK_THRESHOLD,
                         help=f"Mismatch score threshold for the correctness check (default {cfg.NO_EVIDENCE_RISK_THRESHOLD}).")
    args = parser.parse_args()

    global DATASET_PATH, RESULTS_DIR, CACHE_PATH, OUTPUT_CSV
    DATASET_PATH = args.dataset_path
    RESULTS_DIR = args.results_dir
    CACHE_PATH = RESULTS_DIR / "no_evidence_claims_cache.json"
    OUTPUT_CSV = RESULTS_DIR / "no_evidence_results.csv"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_PATH)

    rows = []
    for _, row in df.iterrows():
        ev = parse_github_evidence(row["GitHub_Evidence"])
        if is_no_evidence_row(ev):
            rows.append((row, classify_github_status(ev)))

    print(f"Found {len(rows)} rows where the normal evidence-grounded pipeline "
          f"has nothing to work with (empty or unresolved GitHub).\n")

    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    cache = load_cache()

    results = []
    for row, status in rows:
        h = row_hash(row)
        cache_key = f"{row['ID']}:{h}"
        if cache_key in cache:
            a = ClaimEvidenceAssessment.model_validate(cache[cache_key])
        else:
            a = assess_no_evidence(client, row["Transcript"], row["Resume"], status)
            cache[cache_key] = a.model_dump()
            save_cache(cache)

        correct = int(a.mismatch_score >= args.threshold) == int(row["Is_Fraud"])
        results.append({
            "ID": row["ID"], "Name": row["Name"], "Case_Type": row["Case_Type"],
            "github_status": status, "Is_Fraud": row["Is_Fraud"],
            "mismatch_score": a.mismatch_score, "number_of_contradictions": a.number_of_contradictions,
            "correct_at_threshold": correct,
            "key_contradictions": " | ".join(a.key_contradictions) if a.key_contradictions else "",
        })
        mark = "OK " if correct else "MISS"
        print(f"  [{mark}] {row['Name']:<24} ({status:<10}) {row['Case_Type']:<20} "
              f"Is_Fraud={row['Is_Fraud']}  mismatch={a.mismatch_score:.2f}  "
              f"contradictions={a.number_of_contradictions}")

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_CSV, index=False)

    n_correct = out["correct_at_threshold"].sum()
    print(f"\n{n_correct}/{len(out)} correct at a {args.threshold} mismatch threshold "
          f"(illustrative only — n={len(out)} from {out['Name'].nunique()} real people).")
    print(f"\nSaved: {OUTPUT_CSV}")
    print("\nThis validates the MECHANISM works directionally on known real cases.")
    print("It is not, and cannot be, a validated classifier at this n. To build")
    print("one, you would need many more real candidates with unresolvable or")
    print("empty GitHub profiles and known labels — a fundamentally different,")
    print("smaller population than most real candidates.")


if __name__ == "__main__":
    main()
