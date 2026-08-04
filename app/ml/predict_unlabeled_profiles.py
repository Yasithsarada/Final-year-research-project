"""
Applies the classifier trained in train_on_real_dataset.py (24 real, labelled
rows) to a larger set of REAL people who have no fraud label at all:
datasets/transcripted interview data-filtered.csv (78 unique real GitHub
accounts, résumé/transcript text still LLM-generated, GitHub link real).

Run from the `reasearch v2` project root:
    python -m app.ml.predict_unlabeled_profiles

WHAT THIS DOES (two phases, since GitHub fetch and Gemini calls have very
different concurrency/rate-limit characteristics):
  Phase 1 — concurrent live GitHub fetch (MSRAgent) for every candidate.
  Phase 2 — sequential, paced Gemini claim/evidence assessment (same prompt
            and schema as train_on_real_dataset.py), cached to disk.
  Then: refit the classifier on the 24 labelled rows (from their cache —
  zero new Gemini calls needed there) and predict a fraud probability for
  every candidate here.

READ THIS BEFORE TRUSTING A SINGLE NUMBER: there is no ground truth for this
file. These are predictions, not measurements. Use them to spot-check
against people you actually know — not to report an accuracy figure, because
there is nothing here to measure accuracy against.
"""
import asyncio
import csv
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from google import genai

from app.core.config import settings
import app.ml.pipeline_config as cfg
from app.services.msr_agent import MSRAgent
from app.ml.train_on_real_dataset import (
    FEATURE_NAMES, ClaimEvidenceAssessment, assess_claims,
    generic_message_ratio, parse_github_evidence,
)
from app.ml.no_evidence_assessment import assess_no_evidence

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Module-level defaults, overridable via CLI flags on main() or the
# ML_UNLABELED_DATASET_PATH / ML_LABELED_DATASET_PATH env vars in .env.
LABELED_DATASET_PATH = cfg.LABELED_DATASET_PATH
LABELED_CACHE_PATH = cfg.RESULTS_DIR / "real_dataset_claims_cache.json"
DATASET_PATH = cfg.UNLABELED_DATASET_PATH
RESULTS_DIR = cfg.RESULTS_DIR
OUTPUT_CSV = RESULTS_DIR / "transcripted_dataset_predictions.csv"
CACHE_PATH = RESULTS_DIR / "transcripted_dataset_claims_cache.json"
NO_EVIDENCE_THRESHOLD = cfg.NO_EVIDENCE_RISK_THRESHOLD

GITHUB_URL_RE = re.compile(r"github\.com/([A-Za-z0-9-]+)", re.IGNORECASE)
MAX_CONCURRENT_GITHUB_FETCHES = cfg.MAX_CONCURRENT_GITHUB_FETCHES
GEMINI_MIN_INTERVAL_SECONDS = cfg.GEMINI_MIN_INTERVAL_SECONDS


# ----------------------------------------------------------------------
# Refit the classifier on the 24 labelled rows (reads from its own cache —
# no new Gemini calls needed if it's already fully populated).
# ----------------------------------------------------------------------
def fit_classifiers(labeled_dataset_path: Path = None, labeled_cache_path: Path = None, seed: int = None):
    labeled_dataset_path = labeled_dataset_path or LABELED_DATASET_PATH
    labeled_cache_path = labeled_cache_path or LABELED_CACHE_PATH
    seed = cfg.RANDOM_SEED if seed is None else seed

    df = pd.read_csv(labeled_dataset_path)
    cache = json.loads(labeled_cache_path.read_text(encoding="utf-8")) if labeled_cache_path.exists() else {}

    import hashlib
    rows = []
    for _, row in df.iterrows():
        ev = parse_github_evidence(row["GitHub_Evidence"])
        payload = f"{row['Transcript']}|{row['Resume']}|{row['GitHub_Evidence']}"
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        cache_key = f"{row['ID']}:{h}"
        if cache_key not in cache:
            raise RuntimeError(
                f"Labelled dataset cache is missing row {row['ID']} — run "
                "train_on_real_dataset.py first so the classifier has something to fit."
            )
        a = ClaimEvidenceAssessment.model_validate(cache[cache_key])
        rows.append({
            "total_commits": ev.get("total_commits", 0),
            "average_commit_size_lines": ev.get("average_commit_size_lines", 0.0),
            "total_prs": ev.get("total_prs", 0),
            "approval_rate": ev.get("approval_rate", 0.0),
            "total_repos": ev.get("total_repos", 0),
            "followers": ev.get("followers", 0),
            "num_languages": len(ev.get("primary_languages", {}) or {}),
            "generic_message_ratio": generic_message_ratio(ev.get("sample_commit_messages", [])),
            "mismatch_score": a.mismatch_score,
            "number_of_contradictions": a.number_of_contradictions,
        })

    X_train = pd.DataFrame(rows)[FEATURE_NAMES].to_numpy(dtype=float)
    y_train = df["Is_Fraud"].to_numpy(dtype=int)

    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    logreg.fit(X_train, y_train)
    gbm = GradientBoostingClassifier(
        n_estimators=cfg.REAL_GBM_N_ESTIMATORS, learning_rate=cfg.REAL_GBM_LEARNING_RATE,
        max_depth=cfg.REAL_GBM_MAX_DEPTH, random_state=seed
    )
    gbm.fit(X_train, y_train)
    print(f"Classifiers fit on {len(y_train)} labelled real rows "
          f"({int(y_train.sum())} fraud / {len(y_train) - int(y_train.sum())} genuine).")
    return logreg, gbm


# ----------------------------------------------------------------------
# Load and dedupe the unlabelled candidate list
# ----------------------------------------------------------------------
def load_candidates(dataset_path: Path = None) -> pd.DataFrame:
    dataset_path = dataset_path or DATASET_PATH
    df = pd.read_csv(dataset_path).dropna(how="all")
    df["github_user"] = df["Resume"].apply(
        lambda text: (GITHUB_URL_RE.findall(str(text)) or [None])[0]
    )
    before = len(df)
    df = df.drop_duplicates(subset=["ID", "github_user"], keep="first")
    if len(df) < before:
        print(f"Dropped {before - len(df)} exact duplicate row(s).")
    df = df.dropna(subset=["github_user"])
    return df.reset_index(drop=True)


def gh_to_evidence_dict(gh) -> dict:
    """Maps a live-fetched GitHubProfileMetrics into the same shape as the
    cached GitHub_Evidence JSON in the labelled dataset, so the exact same
    feature-building code applies to both."""
    return {
        "total_commits": gh.commit_history.total_commits,
        "frequency_per_week": gh.commit_history.frequency_per_week,
        "average_commit_size_lines": gh.commit_history.average_commit_size_lines,
        "merging_percentage": gh.commit_history.merging_percentage,
        "total_prs": gh.pull_requests.total_prs,
        "accepted_prs": gh.pull_requests.accepted_prs,
        "approval_rate": gh.pull_requests.approval_rate,
        "total_repos": gh.repository_metadata.total_repos,
        "total_stars": gh.repository_metadata.total_stars,
        "total_forks": gh.repository_metadata.total_forks,
        "followers": gh.repository_metadata.followers,
        "primary_languages": gh.repository_metadata.primary_languages,
        "sample_commit_messages": gh.commit_history.commit_messages[:20],
    }


# ----------------------------------------------------------------------
# Phase 1: concurrent GitHub fetch
# ----------------------------------------------------------------------
async def fetch_all_github(candidates: pd.DataFrame) -> dict:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_GITHUB_FETCHES)
    results = {}
    completed = 0
    total = len(candidates)

    async def fetch_one(row):
        nonlocal completed
        async with semaphore:
            msr = MSRAgent(
                stats_max_retries=cfg.STATS_MAX_RETRIES,
                stats_retry_delay_seconds=cfg.STATS_RETRY_DELAY_SECONDS,
            )
            try:
                gh = await msr.analyze_profile(row["github_user"])
                results[row["ID"]] = ("OK", gh)
            except Exception as e:
                results[row["ID"]] = (f"FETCH_FAILED: {e}", None)
        completed += 1
        status = results[row["ID"]][0]
        print(f"  [GitHub {completed}/{total}] {row['Name']:<26} ({row['github_user']:<20}) -> "
              f"{'OK' if status == 'OK' else status}")

    await asyncio.gather(*(fetch_one(row) for _, row in candidates.iterrows()))
    return results


# ----------------------------------------------------------------------
# Phase 2: sequential, paced, cached Gemini assessment + prediction
# ----------------------------------------------------------------------
def load_cache() -> dict:
    return json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def row_hash(row) -> str:
    import hashlib
    payload = f"{row['Transcript']}|{row['Resume']}|{row['github_user']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def risk_bucket(p: float) -> str:
    if p >= 0.60:
        return "High"
    if p >= 0.30:
        return "Medium"
    return "Low"


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=cfg.UNLABELED_DATASET_PATH,
                         help=f"Unlabeled candidates CSV to score (default {cfg.UNLABELED_DATASET_PATH}). "
                              "Validate first with: python -m app.ml.dataset_validator --unlabeled <path>")
    parser.add_argument("--labeled-dataset-path", type=Path, default=cfg.LABELED_DATASET_PATH,
                         help="The labeled dataset the classifier is trained on "
                              f"(default {cfg.LABELED_DATASET_PATH}). Must already have a populated "
                              "cache (see --labeled-cache-path) — run train_on_real_dataset.py on it first.")
    parser.add_argument("--labeled-cache-path", type=Path, default=LABELED_CACHE_PATH,
                         help=f"Gemini claims cache for --labeled-dataset-path (default {LABELED_CACHE_PATH}).")
    parser.add_argument("--results-dir", type=Path, default=cfg.RESULTS_DIR,
                         help=f"Where to write predictions/cache (default {cfg.RESULTS_DIR}).")
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED,
                         help=f"Random seed for the refit classifiers (default {cfg.RANDOM_SEED}).")
    args = parser.parse_args()

    global DATASET_PATH, RESULTS_DIR, OUTPUT_CSV, CACHE_PATH
    DATASET_PATH = args.dataset_path
    RESULTS_DIR = args.results_dir
    OUTPUT_CSV = RESULTS_DIR / "transcripted_dataset_predictions.csv"
    CACHE_PATH = RESULTS_DIR / "transcripted_dataset_claims_cache.json"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logreg, gbm = fit_classifiers(args.labeled_dataset_path, args.labeled_cache_path, args.seed)

    candidates = load_candidates(DATASET_PATH)
    print(f"\n{len(candidates)} real candidates to score.\n")

    print("=" * 72)
    print("PHASE 1 — LIVE GITHUB FETCH")
    print("=" * 72)
    gh_results = asyncio.run(fetch_all_github(candidates))

    print("\n" + "=" * 72)
    print("PHASE 2 — CLAIM/EVIDENCE ASSESSMENT + PREDICTION")
    print("=" * 72)

    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    cache = load_cache()

    rows_out = []
    all_rows = list(candidates.iterrows())
    for i, (_, row) in enumerate(all_rows):
        status, gh = gh_results.get(row["ID"], ("FAIL", None))
        h = row_hash(row)

        if status == "OK":
            ev = gh_to_evidence_dict(gh)
            is_empty = ev["total_commits"] == 0 and ev["total_repos"] == 0
        else:
            ev = None
            is_empty = False

        if status == "OK" and not is_empty:
            # Normal path: real, non-empty GitHub evidence to check claims against.
            evidence_status = "full"
            cache_key = f"{row['ID']}:{h}"
            if cache_key in cache:
                a = ClaimEvidenceAssessment.model_validate(cache[cache_key])
            else:
                a = assess_claims(client, row["Transcript"], row["Resume"], ev)
                cache[cache_key] = a.model_dump()
                save_cache(cache)
                time.sleep(GEMINI_MIN_INTERVAL_SECONDS)

            feat = {
                "total_commits": ev["total_commits"],
                "average_commit_size_lines": ev["average_commit_size_lines"],
                "total_prs": ev["total_prs"],
                "approval_rate": ev["approval_rate"],
                "total_repos": ev["total_repos"],
                "followers": ev["followers"],
                "num_languages": len(ev["primary_languages"] or {}),
                "generic_message_ratio": generic_message_ratio(ev["sample_commit_messages"]),
                "mismatch_score": a.mismatch_score,
                "number_of_contradictions": a.number_of_contradictions,
            }
            X = pd.DataFrame([feat])[FEATURE_NAMES].to_numpy(dtype=float)
            p_logreg = float(logreg.predict_proba(X)[0, 1])
            p_gbm = float(gbm.predict_proba(X)[0, 1])
            predicted_risk = risk_bucket((p_logreg + p_gbm) / 2)
            logreg_str, gbm_str = f"{p_logreg:.2f}", f"{p_gbm:.2f}"
        else:
            # No usable GitHub evidence — either the fetch failed outright
            # (github_status="unresolved") or it succeeded but is verifiably
            # empty (github_status="empty"). The full classifier was trained
            # on real evidence features these candidates don't have, so it
            # isn't applied here — report the no-evidence mismatch_score
            # directly instead (see no_evidence_assessment.py).
            evidence_status = "empty" if status == "OK" else "unresolved"
            cache_key = f"{row['ID']}:{h}:noev"
            if cache_key in cache:
                a = ClaimEvidenceAssessment.model_validate(cache[cache_key])
            else:
                a = assess_no_evidence(client, row["Transcript"], row["Resume"], evidence_status)
                cache[cache_key] = a.model_dump()
                save_cache(cache)
                time.sleep(GEMINI_MIN_INTERVAL_SECONDS)

            feat = {
                "total_commits": ev["total_commits"] if ev else 0,
                "average_commit_size_lines": ev["average_commit_size_lines"] if ev else 0.0,
                "total_prs": ev["total_prs"] if ev else 0,
                "approval_rate": ev["approval_rate"] if ev else 0.0,
                "total_repos": ev["total_repos"] if ev else 0,
                "followers": ev["followers"] if ev else 0,
                "num_languages": len(ev["primary_languages"] or {}) if ev else 0,
                "generic_message_ratio": 0.0,
                "mismatch_score": a.mismatch_score,
                "number_of_contradictions": a.number_of_contradictions,
            }
            predicted_risk = "High" if a.mismatch_score >= NO_EVIDENCE_THRESHOLD else "Low"
            logreg_str, gbm_str = "n/a", "n/a"

        result = {
            "ID": row["ID"], "Name": row["Name"], "Role": row["Role"], "github_user": row["github_user"],
            "evidence_status": evidence_status,
            **feat,
            "logreg_fraud_proba": logreg_str,
            "gbm_fraud_proba": gbm_str,
            "predicted_risk": predicted_risk,
            "key_contradictions": " | ".join(a.key_contradictions) if a.key_contradictions else "",
        }
        rows_out.append(result)
        print(f"  [{i+1}/{len(all_rows)}] {row['Name']:<26} ({row['github_user']:<20}) [{evidence_status:<10}] -> "
              f"{result['predicted_risk']:<7} (logreg={logreg_str}, gbm={gbm_str}, "
              f"mismatch={a.mismatch_score:.2f})")

    fieldnames = [
        "ID", "Name", "Role", "github_user", "evidence_status", *FEATURE_NAMES,
        "logreg_fraud_proba", "gbm_fraud_proba", "predicted_risk", "key_contradictions",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    risk_counts = {}
    evidence_counts = {}
    for r in rows_out:
        risk_counts[r["predicted_risk"]] = risk_counts.get(r["predicted_risk"], 0) + 1
        evidence_counts[r["evidence_status"]] = evidence_counts.get(r["evidence_status"], 0) + 1

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Scored: {len(rows_out)} (all candidates — no one is dropped anymore)")
    print(f"  Evidence: full={evidence_counts.get('full', 0)}  "
          f"empty={evidence_counts.get('empty', 0)}  unresolved={evidence_counts.get('unresolved', 0)}")
    for level in ["Low", "Medium", "High"]:
        if level in risk_counts:
            print(f"  {level:<8} {risk_counts[level]}")
    print(f"\nSaved: {OUTPUT_CSV}")
    print("\nREMINDER: there is no ground-truth label for this dataset. These are")
    print("predictions from a classifier trained on only 24 real rows — treat them")
    print("as a starting point for manual review, not a measured accuracy.")


if __name__ == "__main__":
    main()
