"""
Builds a self-contained, Colab-ready zip of the ML pipeline — the exact
files needed to run every script under app/ml/, nothing else (no FastAPI,
no spaCy, no faster-whisper). You run this yourself, any time, so you are
never dependent on anyone regenerating it for you.

Run from the `reasearch v2` project root:
    python scripts/build_colab_package.py

Produces: colab_fraud_pipeline_package.zip (project root)

What's included:
  - Every file under app/ml/
  - The app/services/* and app/models/* modules the ML scripts actually
    import (verified by tracing real imports, not guessed — see the import
    trace in app/ml/README.md "What's in the Colab package")
  - app/core/{config.py, exceptions.py} and every __init__.py needed to make
    these importable as the `app` package
  - requirements-ml.txt
  - datasets/interview_cv_github_fraud_dataset.csv (labeled) and
    datasets/transcripted interview data-filtered.csv (unlabeled) — the two
    datasets the pipeline actually reads. NOT the 74MB original corpus.

What's NOT included (by design — pass --with-app-data to add it):
  - app_data/ (trained model, cached Gemini responses, prior results) — a
    fresh Colab run should train from scratch; if you want to carry over a
    trained model or a warm claims cache instead of re-spending API quota,
    re-run with --with-app-data.
  - Your .env (contains secrets) — Colab instructions use getpass instead.
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Every file the ML pipeline needs, relative to PROJECT_ROOT. Kept as an
# explicit list (not a directory copy) so it's obvious exactly what ships —
# add a file here if you add a new app/ml/ script that imports something new.
FILES = [
    "app/__init__.py",
    "app/core/__init__.py",
    "app/core/config.py",
    "app/core/exceptions.py",
    "app/models/__init__.py",
    "app/models/resume.py",
    "app/models/ccs.py",
    "app/models/github_metrics.py",
    "app/models/evaluation.py",
    "app/services/__init__.py",
    "app/services/anomaly_model.py",
    "app/services/feature_engineering.py",
    "app/services/msr_agent.py",
    "app/services/decision_engine.py",
    "app/services/sca_agent.py",
    "app/services/llm_client.py",
    "app/ml/__init__.py",
    "app/ml/pipeline_config.py",
    "app/ml/dataset_validator.py",
    "app/ml/synthetic_data.py",
    "app/ml/train_anomaly_model.py",
    "app/ml/train_on_real_dataset.py",
    "app/ml/predict_unlabeled_profiles.py",
    "app/ml/evaluate_no_evidence_model.py",
    "app/ml/evaluate_real_profiles.py",
    "app/ml/no_evidence_assessment.py",
    "requirements-ml.txt",
]

DATASETS = [
    "datasets/interview_cv_github_fraud_dataset.csv",
    "datasets/transcripted interview data-filtered.csv",
]

APP_DATA_DIRS = ["app_data/models", "app_data/ml_results"]


def build(output_path: Path, include_app_data: bool, include_datasets: bool):
    missing = [f for f in FILES if not (PROJECT_ROOT / f).exists()]
    if missing:
        print("ERROR: these expected files are missing — the file list in this "
              "script is out of date with the project:", file=sys.stderr)
        for f in missing:
            print(f"  {f}", file=sys.stderr)
        sys.exit(1)

    if output_path.exists():
        output_path.unlink()

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel_path in FILES:
            zf.write(PROJECT_ROOT / rel_path, rel_path)

        if include_datasets:
            for rel_path in DATASETS:
                src = PROJECT_ROOT / rel_path
                if src.exists():
                    zf.write(src, rel_path)
                else:
                    print(f"WARNING: dataset not found, skipped: {rel_path}", file=sys.stderr)

        if include_app_data:
            for rel_dir in APP_DATA_DIRS:
                src_dir = PROJECT_ROOT / rel_dir
                if not src_dir.exists():
                    continue
                for p in src_dir.rglob("*"):
                    if p.is_file():
                        zf.write(p, p.relative_to(PROJECT_ROOT).as_posix())

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Built: {output_path}  ({size_mb:.1f} MB)")
    print(f"  {len(FILES)} code files"
          + (f" + {len(DATASETS)} dataset(s)" if include_datasets else "")
          + (f" + app_data/ (model + results + caches)" if include_app_data else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path,
                         default=PROJECT_ROOT / "colab_fraud_pipeline_package.zip",
                         help="Output zip path (default: colab_fraud_pipeline_package.zip in project root).")
    parser.add_argument("--no-datasets", action="store_true",
                         help="Exclude the two datasets (e.g. if you'll upload your own in Colab).")
    parser.add_argument("--with-app-data", action="store_true",
                         help="Also include app_data/models and app_data/ml_results — carries over "
                              "the trained model and cached Gemini responses, so a Colab run resumes "
                              "instead of starting from scratch and re-spending API quota.")
    args = parser.parse_args()

    build(args.output, include_app_data=args.with_app_data, include_datasets=not args.no_datasets)


if __name__ == "__main__":
    main()
