"""
Validates a candidate CSV against the schema the pipeline actually expects,
BEFORE you spend Gemini calls / GitHub API quota on a run that will fail
partway through on a KeyError. Two schemas are supported — see
app/ml/README.md "Dataset formats" for the full column-by-column spec.

Run directly to validate a file:
    python -m app.ml.dataset_validator --labeled path/to/your_dataset.csv
    python -m app.ml.dataset_validator --unlabeled path/to/your_dataset.csv
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

GITHUB_URL_RE = re.compile(r"github\.com/([A-Za-z0-9-]+)", re.IGNORECASE)

LABELED_REQUIRED_COLUMNS = [
    "ID", "Name", "Role", "Transcript", "Resume",
    "GitHub_URL", "GitHub_Evidence", "Case_Type", "Is_Fraud",
]
UNLABELED_REQUIRED_COLUMNS = ["ID", "Name", "Role", "Transcript", "Resume"]


class DatasetValidationError(Exception):
    pass


def _check_columns(df: pd.DataFrame, required: list[str], path: Path):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DatasetValidationError(
            f"{path} is missing required column(s): {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            f"Required columns: {required}"
        )


def validate_labeled_dataset(path: Path) -> dict:
    """Validates a dataset intended for train_on_real_dataset.py.

    Required columns:
      ID              - unique row identifier (str)
      Name            - candidate name (str)
      Role            - job role/title for this interview case (str)
      Transcript      - interview dialogue text (str)
      Resume          - resume/CV text (str)
      GitHub_URL      - the candidate's GitHub profile URL (str)
      GitHub_Evidence - JSON string of real, fetched GitHub metrics — see
                        app/ml/train_on_real_dataset.py:parse_github_evidence
                        for the exact schema. Must be real evidence (fetched
                        via MSRAgent), not fabricated, or the classifier
                        trains on garbage.
      Is_Fraud        - ground-truth label, 0 (genuine) or 1 (fraud/padded)
      Case_Type       - free-text label for your own bookkeeping (e.g.
                        "strong_alignment"); not read by the pipeline logic,
                        only used in printed progress lines.

    Returns a dict of basic stats for a human sanity-check.
    """
    df = pd.read_csv(path)
    _check_columns(df, LABELED_REQUIRED_COLUMNS, path)

    if df["Is_Fraud"].isna().any():
        raise DatasetValidationError(f"{path}: Is_Fraud has missing values.")
    bad_labels = set(df["Is_Fraud"].unique()) - {0, 1}
    if bad_labels:
        raise DatasetValidationError(
            f"{path}: Is_Fraud must be 0 or 1, found: {bad_labels}"
        )

    for col in ["Transcript", "Resume", "GitHub_Evidence"]:
        if df[col].isna().any() or (df[col].astype(str).str.strip() == "").any():
            raise DatasetValidationError(f"{path}: column '{col}' has empty/missing rows.")

    n_fraud = int(df["Is_Fraud"].sum())
    n_total = len(df)
    if n_fraud == 0 or n_fraud == n_total:
        raise DatasetValidationError(
            f"{path}: Is_Fraud is not both classes present ({n_fraud}/{n_total} "
            "are fraud) — cross-validation needs both classes."
        )
    if min(n_fraud, n_total - n_fraud) < 4:
        print(
            f"WARNING: minority class has only {min(n_fraud, n_total - n_fraud)} "
            "rows. N_FOLDS in pipeline_config.py must be <= this number.",
            file=sys.stderr,
        )

    dupes = df["ID"].duplicated().sum()
    if dupes:
        print(f"WARNING: {dupes} duplicate ID value(s) — rows will still process "
              "but this usually indicates a copy-paste mistake.", file=sys.stderr)

    return {
        "path": str(path), "n_rows": n_total, "n_fraud": n_fraud,
        "n_genuine": n_total - n_fraud, "n_unique_people": df["Name"].nunique(),
        "duplicate_ids": int(dupes),
    }


def validate_unlabeled_dataset(path: Path) -> dict:
    """Validates a dataset intended for predict_unlabeled_profiles.py.

    Required columns:
      ID, Name, Role, Transcript, Resume - same meaning as the labeled
      schema, but NO Is_Fraud/GitHub_Evidence/GitHub_URL required — the
      GitHub link is extracted from a `github.com/<username>` pattern
      found anywhere in the Resume text, and evidence is fetched live.

    Returns a dict of basic stats, including how many rows actually have
    an extractable GitHub link (rows without one are skipped at runtime).
    """
    df = pd.read_csv(path).dropna(how="all")
    _check_columns(df, UNLABELED_REQUIRED_COLUMNS, path)

    for col in ["Transcript", "Resume"]:
        if df[col].isna().any() or (df[col].astype(str).str.strip() == "").any():
            raise DatasetValidationError(f"{path}: column '{col}' has empty/missing rows.")

    github_user = df["Resume"].apply(lambda t: (GITHUB_URL_RE.findall(str(t)) or [None])[0])
    n_with_link = int(github_user.notna().sum())
    if n_with_link == 0:
        raise DatasetValidationError(
            f"{path}: no row's Resume text contains a github.com/<username> "
            "link — predict_unlabeled_profiles.py has nothing to fetch."
        )

    dupes = df["ID"].duplicated().sum()
    return {
        "path": str(path), "n_rows": len(df), "n_with_github_link": n_with_link,
        "n_without_github_link": len(df) - n_with_link, "duplicate_ids": int(dupes),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--labeled", type=str, help="Path to a labeled (Is_Fraud) dataset to validate.")
    group.add_argument("--unlabeled", type=str, help="Path to an unlabeled dataset to validate.")
    args = parser.parse_args()

    try:
        if args.labeled:
            stats = validate_labeled_dataset(Path(args.labeled))
        else:
            stats = validate_unlabeled_dataset(Path(args.unlabeled))
    except DatasetValidationError as e:
        print(f"INVALID: {e}", file=sys.stderr)
        sys.exit(1)

    print("VALID —", stats["path"])
    for k, v in stats.items():
        if k != "path":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
