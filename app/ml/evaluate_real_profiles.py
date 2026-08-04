"""
Real-world validation harness for the Fraud & Anomaly Detection Model.

Run from the `reasearch v2` project root:
    python -m app.ml.evaluate_real_profiles

What this does:
  1. Loads datasets/transcripted interview data-filtered.csv — a small,
     hand-curated set of candidates whose résumé text is still LLM-generated,
     but whose embedded GitHub link is a REAL, verifiable account.
  2. Extracts the GitHub username from each résumé and fetches REAL profile
     data via the authenticated MSRAgent (same code path used in production).
  3. Runs each real profile through the REAL FeatureEngineeringLayer and
     DecisionEngine — including the IsolationForest trained by
     train_anomaly_model.py — exactly as the live API would.
  4. Saves one row per candidate to app_data/ml_results/real_profile_evaluation.csv
     and prints a readable table.

READ THIS BEFORE QUOTING NUMBERS: there are only a handful of candidates here
and no ground-truth fraud label — this is a case-study / spot-check set, not
a validation set. Judge each verdict yourself from what you know about these
people; do not compute or report an accuracy/precision figure from this file.
Cross-modal signals (skill_mismatch, audio_resume_mismatch) are left at 0 in
this pass since the résumé text is not parsed into structured claims here —
only the GitHub-evidence path is real.
"""
import argparse
import asyncio
import csv
import logging
import re
from pathlib import Path

import pandas as pd

import app.ml.pipeline_config as cfg
from app.services.msr_agent import MSRAgent
from app.services.feature_engineering import FeatureEngineeringLayer
from app.services.sca_agent import SCAAgent
from app.services.decision_engine import DecisionEngine

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DATASET_PATH = cfg.UNLABELED_DATASET_PATH
RESULTS_DIR = cfg.RESULTS_DIR
OUTPUT_CSV = RESULTS_DIR / "real_profile_evaluation.csv"

GITHUB_URL_RE = re.compile(r"github\.com/([A-Za-z0-9-]+)", re.IGNORECASE)


def load_candidates(dataset_path: Path = None) -> pd.DataFrame:
    """Loads the filtered CSV, drops blank rows and the one exact duplicate,
    and extracts the real GitHub username embedded in each résumé."""
    dataset_path = dataset_path or DATASET_PATH
    df = pd.read_csv(dataset_path).dropna(how="all")
    df["github_user"] = df["Resume"].apply(
        lambda text: (GITHUB_URL_RE.findall(str(text)) or [None])[0]
    )
    before = len(df)
    df = df.drop_duplicates(subset=["ID", "github_user"], keep="first")
    if len(df) < before:
        logger.info(f"Dropped {before - len(df)} exact duplicate row(s).")

    # Same synthetic identity, two different real GitHub accounts — not a
    # duplicate, but ambiguous. Surface it rather than silently picking one.
    dupe_names = df[df.duplicated(subset=["Name"], keep=False)]
    if not dupe_names.empty:
        for name, group in dupe_names.groupby("Name"):
            users = group["github_user"].tolist()
            print(
                f"NOTE: '{name}' is linked to {len(users)} different GitHub "
                f"accounts in this dataset ({', '.join(users)}). Both are "
                "evaluated separately below — decide which (if either) is "
                "authoritative for your write-up."
            )

    missing = df[df["github_user"].isna()]
    if not missing.empty:
        print(f"WARNING: no GitHub link found for {len(missing)} row(s); skipping them.")
        df = df.dropna(subset=["github_user"])

    return df.reset_index(drop=True)


async def evaluate_candidate(row, fel: FeatureEngineeringLayer, de: DecisionEngine, cq) -> dict:
    """Runs one real candidate through the real production pipeline."""
    msr = MSRAgent(
        stats_max_retries=cfg.STATS_MAX_RETRIES,
        stats_retry_delay_seconds=cfg.STATS_RETRY_DELAY_SECONDS,
    )
    try:
        gh = await msr.analyze_profile(row["github_user"])
    except Exception as e:
        logger.warning(f"Could not fetch GitHub data for '{row['github_user']}': {e}")
        return {
            "ID": row["ID"],
            "Name": row["Name"],
            "Role": row["Role"],
            "github_user": row["github_user"],
            "status": f"FETCH_FAILED: {e}",
        }

    fi = fel.compute_fraud_indicators(gh)
    tse = fel.compute_tse(gh, cq)
    nts = fel.compute_nts(gh)
    result = de.evaluate(tse, nts, fi, gh)

    return {
        "ID": row["ID"],
        "Name": row["Name"],
        "Role": row["Role"],
        "github_user": row["github_user"],
        "status": "OK",
        "total_commits": gh.commit_history.total_commits,
        "total_repos": gh.repository_metadata.total_repos,
        "followers": gh.repository_metadata.followers,
        "total_stars": gh.repository_metadata.total_stars,
        "burst_concentration_ratio": fi.burst_concentration_ratio,
        "commit_coefficient_variation": fi.commit_coefficient_variation,
        "generic_message_ratio": fi.generic_message_ratio,
        "pr_rejection_rate": fi.pr_rejection_rate,
        "flags": " | ".join(fi.flags) if fi.flags else "",
        "technical_skill_score": result.technical_skill_score,
        "collaboration_consistency_score": result.collaboration_consistency_score,
        "fraud_risk_level": result.fraud_risk_level,
        "fraud_index": result.fraud_index,
        "ml_anomaly_score": result.ml_anomaly_score,
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=cfg.UNLABELED_DATASET_PATH,
                         help=f"Candidates CSV to evaluate (default {cfg.UNLABELED_DATASET_PATH}). "
                              "Validate first with: python -m app.ml.dataset_validator --unlabeled <path>")
    parser.add_argument("--results-dir", type=Path, default=cfg.RESULTS_DIR,
                         help=f"Where to write results (default {cfg.RESULTS_DIR}).")
    args = parser.parse_args()

    global RESULTS_DIR, OUTPUT_CSV
    RESULTS_DIR = args.results_dir
    OUTPUT_CSV = RESULTS_DIR / "real_profile_evaluation.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(args.dataset_path)
    print(f"\nEvaluating {len(candidates)} real GitHub profile(s)...\n")

    fel = FeatureEngineeringLayer()
    de = DecisionEngine()
    cq = SCAAgent._mock_metrics()  # same fallback the live API uses when SonarQube is offline

    rows = []
    for _, row in candidates.iterrows():
        result = await evaluate_candidate(row, fel, de, cq)
        rows.append(result)
        status = result["status"]
        if status == "OK":
            print(
                f"  {result['Name']:<26} ({result['github_user']:<20}) -> "
                f"{result['fraud_risk_level']:<7} "
                f"(fraud_index={result['fraud_index']}, "
                f"ml_anomaly_score={result['ml_anomaly_score']})"
            )
        else:
            print(f"  {result['Name']:<26} ({result['github_user']:<20}) -> {status}")

    fieldnames = [
        "ID", "Name", "Role", "github_user", "status",
        "total_commits", "total_repos", "followers", "total_stars",
        "burst_concentration_ratio", "commit_coefficient_variation",
        "generic_message_ratio", "pr_rejection_rate", "flags",
        "technical_skill_score", "collaboration_consistency_score",
        "fraud_risk_level", "fraud_index", "ml_anomaly_score",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\nSaved: {OUTPUT_CSV}")
    print(
        "\nREMINDER: this is a small, unlabelled, real-world case-study set "
        "(n={}). Use it for qualitative spot-checking against what you "
        "personally know about these candidates — not as an accuracy "
        "measurement.".format(len(rows))
    )


if __name__ == "__main__":
    asyncio.run(main())
