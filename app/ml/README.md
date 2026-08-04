# Fraud & Anomaly Detection — ML Pipeline

This is the complete, standalone training/evaluation pipeline for the fraud
detection model. It is dataset-independent: every path, seed, and
hyperparameter lives in one file (`pipeline_config.py`), and every script
accepts CLI flags to override those defaults per run — you never need to
edit script internals to point at a different dataset.

This README assumes you have never run the project before.

---

## 1. Requirements

- **Python 3.11** (what this was built and tested on; 3.10+ should work,
  3.9+ is required for the type hints used in `dataset_validator.py`).
- Install dependencies:
  ```bash
  pip install -r requirements-ml.txt
  ```
  This is a **strict subset** of the full application's `requirements.txt`
  — no FastAPI, no spaCy, no faster-whisper, no database drivers. Verified
  by actually tracing every import across all 6 entrypoint scripts (see
  §9 "What's in the Colab package" for how). If you're running this from
  inside the full `reasearch v2` project (not the standalone Colab
  package), the full `requirements.txt` already satisfies this — you don't
  need to install anything extra.

- **API keys** (only required for the scripts that call them — see §3):
  - `GEMINI_API_KEY` — Google AI Studio key, used for claim/evidence
    assessment (`train_on_real_dataset.py`, `predict_unlabeled_profiles.py`,
    `evaluate_no_evidence_model.py`).
  - `GITHUB_TOKEN` — a GitHub personal access token (no special scopes
    needed, just raises your rate limit from 60 to 5,000 req/hr), used for
    live GitHub fetches (`predict_unlabeled_profiles.py`,
    `evaluate_real_profiles.py`).
  - Neither is required for `train_anomaly_model.py` (fully synthetic, no
    external calls).

  Set these in a `.env` file at the project root:
  ```
  GEMINI_API_KEY=your-key-here
  GITHUB_TOKEN=your-token-here
  ```
  In Colab, don't commit a `.env` — see §8 for the `getpass` pattern.

---

## 2. Folder structure

```
reasearch v2/
├─ app/
│  ├─ ml/                          <- THIS pipeline
│  │  ├─ pipeline_config.py        <- single source of truth for all settings
│  │  ├─ dataset_validator.py      <- validate a CSV before spending API quota on it
│  │  ├─ synthetic_data.py         <- synthetic candidate generator
│  │  ├─ train_anomaly_model.py    <- train on synthetic data (no API keys needed)
│  │  ├─ train_on_real_dataset.py  <- train/CV on real labeled data (needs GEMINI_API_KEY)
│  │  ├─ predict_unlabeled_profiles.py  <- apply trained classifier to new, unlabeled real people
│  │  ├─ evaluate_no_evidence_model.py  <- validate the no-GitHub-evidence fallback path
│  │  ├─ evaluate_real_profiles.py      <- GitHub-evidence-only spot-check (no Gemini calls)
│  │  ├─ no_evidence_assessment.py      <- LLM prompts for empty/unresolved GitHub cases
│  │  └─ README.md                 <- this file
│  ├─ services/                    <- production code the pipeline reuses (NOT duplicated)
│  │  ├─ anomaly_model.py          <- IsolationForest wrapper (the production model)
│  │  ├─ feature_engineering.py    <- turns raw GitHub data into the 9-feature vector
│  │  ├─ msr_agent.py              <- live GitHub API client
│  │  ├─ decision_engine.py        <- combines rule engine + ML score
│  │  ├─ sca_agent.py              <- code-quality metrics (mocked when SonarQube is offline)
│  │  └─ llm_client.py             <- shared Gemini/OpenAI schema helper
│  ├─ models/                      <- Pydantic schemas (GitHubProfileMetrics, etc.)
│  └─ core/
│     └─ config.py                 <- app-wide settings (API keys, BASE_DIR)
├─ datasets/
│  ├─ interview_cv_github_fraud_dataset.csv       <- LABELED (has Is_Fraud)
│  └─ transcripted interview data-filtered.csv    <- UNLABELED (no Is_Fraud)
├─ app_data/
│  ├─ models/isolation_forest.joblib   <- the persisted, production-loaded model
│  └─ ml_results/                      <- every script's metrics/figures/caches land here
├─ requirements-ml.txt             <- minimal deps for this pipeline only
└─ scripts/build_colab_package.py  <- regenerate the Colab zip yourself, any time
```

---

## 3. The scripts, what they do, and in what order to run them

| # | Script | Needs API keys? | Reads | Writes |
|---|---|---|---|---|
| 1 | `train_anomaly_model.py` | No | nothing (generates synthetic data) | `isolation_forest.joblib` (production model) + `metrics.json`, `dataset.csv`, `loss_curve.{png,csv}`, `roc_pr_curves.csv`, `pr_curve.csv` |
| 2 | `train_on_real_dataset.py` | `GEMINI_API_KEY` | `LABELED_DATASET_PATH` | `real_dataset_metrics.json`, `real_dataset_features.csv`, `real_dataset_loss_curve.{png,csv}`, `real_dataset_claims_cache.json` |
| 3 | `predict_unlabeled_profiles.py` | `GEMINI_API_KEY` + `GITHUB_TOKEN` | `UNLABELED_DATASET_PATH` (+ the cache from step 2) | `transcripted_dataset_predictions.csv`, `transcripted_dataset_claims_cache.json` |
| 4 | `evaluate_no_evidence_model.py` | `GEMINI_API_KEY` | `LABELED_DATASET_PATH` | `no_evidence_results.csv`, `no_evidence_claims_cache.json` |
| 5 | `evaluate_real_profiles.py` | `GITHUB_TOKEN` only | `UNLABELED_DATASET_PATH` | `real_profile_evaluation.csv` |

**Typical order for a from-scratch run:**
```bash
python -m app.ml.train_anomaly_model              # 1. production model, no keys needed
python -m app.ml.train_on_real_dataset             # 2. real-data validation, needs GEMINI_API_KEY
python -m app.ml.predict_unlabeled_profiles        # 3. apply classifier to new real people
```
Steps 4 and 5 are optional diagnostics — run them if you want the no-evidence
fallback validated, or a fast GitHub-only spot-check without spending Gemini
quota.

Every script accepts `--help` for its exact flags.

---

## 4. `pipeline_config.py` — every configurable value, in one place

Open `app/ml/pipeline_config.py` directly — it's a plain Python file with
every setting documented inline, organised into these sections:

- **Paths** — `LABELED_DATASET_PATH`, `UNLABELED_DATASET_PATH`,
  `RESULTS_DIR`, `MODEL_DIR`, `ISOLATION_FOREST_PATH`.
- **Reproducibility** — `RANDOM_SEED`.
- **Synthetic data generation** — `SYNTHETIC_N_GENUINE`, `SYNTHETIC_N_FRAUD`,
  `TRAIN_TEST_SPLIT`.
- **Hybrid rule/ML weighting** — `HYBRID_RULE_WEIGHT`, `HYBRID_ML_WEIGHT`.
- **Supervised baseline hyperparameters** — `RF_N_ESTIMATORS`, `GBM_*`,
  `LOSS_CURVE_*` (synthetic script) and `REAL_RF_*`, `REAL_GBM_*`,
  `REAL_LOSS_CURVE_*` (real-data script — deliberately shallower models,
  since n≈34 overfits fast with the synthetic script's larger ones).
- **Cross-validation** — `N_FOLDS`.
- **Gemini settings** — `GEMINI_MODEL`, `GEMINI_MIN_INTERVAL_SECONDS`,
  `GEMINI_MAX_RETRIES`.
- **GitHub fetch settings** — `MAX_CONCURRENT_GITHUB_FETCHES`,
  `STATS_MAX_RETRIES`, `STATS_RETRY_DELAY_SECONDS`.
- **No-evidence threshold** — `NO_EVIDENCE_RISK_THRESHOLD`.

Run `python -m app.ml.pipeline_config` any time to print every effective
value before a long run.

### Three ways to change a value (highest priority first)

1. **A CLI flag on the script you're running.** Every script accepts
   `--dataset-path`, `--results-dir`, and `--seed` at minimum; some accept
   more (`--n-genuine`/`--n-fraud`, `--n-folds`, `--threshold`,
   `--model-path`). Run any script with `--help` to see its exact flags.
2. **An environment variable / `.env` entry, prefixed `ML_`.** e.g. add to
   `.env`:
   ```
   ML_RANDOM_SEED=7
   ML_LABELED_DATASET_PATH=datasets/my_other_dataset.csv
   ML_SYNTHETIC_N_GENUINE=2000
   ```
   Every constant name in `pipeline_config.py` works this way — prefix it
   with `ML_`.
3. **Edit `pipeline_config.py` directly.** The defaults live there in plain
   sight, one per line, each with a comment.

---

## 5. Dataset formats

Validate a file **before** running a training/evaluation script on it — this
catches a missing column in one second instead of after several minutes of
GitHub/Gemini calls:
```bash
python -m app.ml.dataset_validator --labeled path/to/your_dataset.csv
python -m app.ml.dataset_validator --unlabeled path/to/your_dataset.csv
```

### Labeled dataset (for `train_on_real_dataset.py`)

Required columns:

| Column | Type | Meaning |
|---|---|---|
| `ID` | str | Unique row identifier |
| `Name` | str | Candidate name |
| `Role` | str | Job role/title for this interview case |
| `Transcript` | str | Interview dialogue text |
| `Resume` | str | Résumé/CV text |
| `GitHub_URL` | str | Candidate's GitHub profile URL |
| `GitHub_Evidence` | str | JSON string of **real, fetched** GitHub metrics (see `parse_github_evidence()` in `train_on_real_dataset.py` for the exact schema) |
| `Is_Fraud` | int | Ground-truth label: `0` = genuine, `1` = fraud/padded |
| `Case_Type` | str | Free-text bookkeeping label (not read by pipeline logic) |

Requirements enforced by the validator: both classes present, minority class
≥ 4 rows (ideally ≥ `N_FOLDS`), no empty `Transcript`/`Resume`/`GitHub_Evidence`.

**On `GitHub_Evidence` being real, not fabricated:** this column must come
from an actual GitHub API fetch (via `MSRAgent.analyze_profile()`), not
invented data — the entire point of this dataset is grounding claims against
verifiable evidence. Fabricating this column would defeat the method.

### Unlabeled dataset (for `predict_unlabeled_profiles.py` / `evaluate_real_profiles.py`)

Required columns: `ID`, `Name`, `Role`, `Transcript`, `Resume`. No
`Is_Fraud`, no `GitHub_URL`/`GitHub_Evidence` column — the GitHub username is
extracted from a `github.com/<username>` pattern found **anywhere in the
`Resume` text**, and evidence is fetched live at run time.

---

## 6. Training from scratch on a new dataset

1. Point `pipeline_config.py` (or `.env`, or a CLI flag) at your file.
2. Validate it: `python -m app.ml.dataset_validator --labeled your_file.csv`
3. Run: `python -m app.ml.train_on_real_dataset --dataset-path your_file.csv`

The Gemini claims cache (`real_dataset_claims_cache.json`) is keyed by row
ID + a content hash, so if you add new rows to the same file the script only
spends API calls on the new/changed rows — everything else replays from
cache instantly.

### Growing the real dataset

The current real-data hyperparameters (`REAL_RF_*`, `REAL_GBM_*` in
`pipeline_config.py`) are deliberately shallow because n≈34 overfits fast.
As your labeled dataset grows past a few hundred rows, raise `REAL_RF_MAX_DEPTH`
and `REAL_GBM_N_ESTIMATORS` back toward the synthetic script's values, and
raise `N_FOLDS` if your minority class supports it.

---

## 7. Evaluating an already-trained model on different data (no retraining)

`predict_unlabeled_profiles.py` does exactly this: it **refits** the
classifier from the labeled dataset's cache (fast, no retraining cost — see
§12 on why "refit" here doesn't mean "expensive") and scores a *different*,
unlabeled dataset:
```bash
python -m app.ml.predict_unlabeled_profiles \
  --labeled-dataset-path datasets/interview_cv_github_fraud_dataset.csv \
  --dataset-path path/to/new_unlabeled_candidates.csv
```
If you specifically want to reuse the exact IsolationForest saved to disk
(rather than refitting), load it directly:
```python
from app.services.anomaly_model import MLAnomalyDetector
detector = MLAnomalyDetector()  # loads app_data/models/isolation_forest.joblib
score = detector.score(feature_vector)  # feature_vector: 9-element array, see FEATURE_NAMES
```

---

## 8. Retraining with a modified dataset and comparing runs

Every script's `--results-dir` flag lets you keep multiple runs side by side
instead of overwriting the previous one:
```bash
python -m app.ml.train_anomaly_model --results-dir app_data/ml_results/run_baseline
# ... edit synthetic_data.py or your dataset ...
python -m app.ml.train_anomaly_model --results-dir app_data/ml_results/run_v2
```
Then diff `metrics.json` between the two directories, or load both
`dataset.csv` files into pandas for a side-by-side feature comparison.

**Note:** `--results-dir` does not change where the production model gets
saved (`app_data/models/isolation_forest.joblib` by default) — pass
`--model-path` too if you don't want a comparison run to overwrite the
model the live application actually loads.

---

## 9. Model save/load

The only model persisted to disk is the production `IsolationForest`,
saved by `train_anomaly_model.py` via `MLAnomalyDetector.save()` to
`app_data/models/isolation_forest.joblib` (joblib format — a pickle-based
serialization, standard for scikit-learn). Every other model in this
pipeline (LogReg/RF/GBM baselines, the loss-curve diagnostic model) is
fit fresh in memory each run and intentionally **not** saved — they exist to
produce a comparison number or a chart, not to be deployed.

To load the saved model for inference:
```python
from app.services.anomaly_model import MLAnomalyDetector
detector = MLAnomalyDetector()          # loads the default path automatically
# or: MLAnomalyDetector(model_path=Path("some/other/isolation_forest.joblib"))
score = detector.score(feature_vector)  # -> float anomaly score
is_anomaly = detector.predict_is_anomaly(feature_vector)  # -> bool
```
This is exactly what `DecisionEngine` does in production — see
`app/services/decision_engine.py`.

To retrain and overwrite it: re-run `train_anomaly_model.py` (default
`--model-path` behaviour), or point `--model-path` elsewhere to keep the old
one.

---

## 10. Reproducibility

Every script accepts `--seed` (default from `pipeline_config.RANDOM_SEED`,
`42`). The same seed **and the same input data** produce the same train/test
splits and the same model — verified directly: re-running
`train_anomaly_model.py` with identical arguments reproduces the exact same
`metrics.json` values (down to the 4th decimal place) and the exact same
`best_iteration` on the loss curve.

What reproducibility does **not** cover:
- **Different machines/OS/BLAS backends** can introduce tiny floating-point
  differences in scikit-learn's tree-building — usually invisible in the
  4-decimal metrics, but not bitwise-guaranteed.
- **Gemini's claim/evidence assessment** is not seeded — the LLM call is
  not deterministic, so `mismatch_score` for a *new* row may vary slightly
  between runs. This is why results are cached per row: once a score is
  computed and cached, every subsequent run reuses it rather than
  re-querying, so *your* reported numbers stay fixed even though the
  underlying model isn't perfectly deterministic.
- **Live GitHub data** changes over time (new commits, new repos) — a
  candidate's evidence fetched today may differ from evidence fetched a
  year from now. This is real-world data, not a reproducibility bug.

---

## 11. Running in Google Colab

Regenerate the self-contained package yourself, any time (don't rely on
having a stale copy someone else made you):
```bash
python scripts/build_colab_package.py
```
This produces `colab_fraud_pipeline_package.zip` at the project root. It
contains exactly the files listed in the script's `FILES` constant — open
`scripts/build_colab_package.py` to see the literal list, nothing is hidden
or auto-discovered. Add `--with-app-data` to also carry over the trained
model and cached Gemini responses (recommended if you don't want a Colab run
to re-spend API quota on rows already cached locally).

**In Colab:**
```python
# 1. Upload colab_fraud_pipeline_package.zip via the file browser, then:
!unzip -oq colab_fraud_pipeline_package.zip -d /content
!pip install -q -r /content/requirements-ml.txt
%cd /content

# 2. Provide API keys securely (never paste them directly into a cell)
import os, getpass
os.environ["GEMINI_API_KEY"] = getpass.getpass("Gemini API key: ")
os.environ["GITHUB_TOKEN"] = getpass.getpass("GitHub token: ")

# 3. Run any script exactly as documented above
!python -m app.ml.train_anomaly_model
!python -m app.ml.train_on_real_dataset
```
To use a different dataset in Colab later, drag-and-drop the new CSV into
`/content/datasets/` (or anywhere) and pass `--dataset-path` — no need to
re-upload the whole zip.

### What's in the Colab package (how the file list was verified)

The file list in `scripts/build_colab_package.py` was not guessed — it was
produced by actually importing all 6 entrypoint scripts and tracing
`sys.modules` before/after to see which `app.*` submodules got pulled in,
then cross-checked against a full extract-and-reimport test in an isolated
temp directory (confirming `BASE_DIR` and every import resolve correctly
with zero dependency on the original project path). If you add a new
`app/ml/*.py` script that imports something not already in the `FILES` list,
`build_colab_package.py` will still run, but the new import will fail in
Colab — add the missing file to the list.

---

## 12. Common errors and how to fix them

| Error | Cause | Fix |
|---|---|---|
| `GEMINI_API_KEY is not set` | `.env` missing the key, or Colab `getpass` step skipped | Set it (§1 / §11) |
| `404 ... no longer available to new users` on a Gemini call | The `GEMINI_MODEL` in config isn't available on your key/project | Try `gemini-flash-lite-latest` (the current default) or list available models: `client.models.list()` and pick one; override via `ML_GEMINI_MODEL` in `.env` |
| `429 RESOURCE_EXHAUSTED` from Gemini | Free-tier rate/quota limit hit | The scripts already back off and retry (`GEMINI_MAX_RETRIES`); if it still fails, wait for the quota window to reset or raise `GEMINI_MIN_INTERVAL_SECONDS` |
| `GitHub API error ... 404` for a specific username | The GitHub handle in that résumé doesn't resolve (typo, deleted account) | Expected and handled — `predict_unlabeled_profiles.py` scores it via the no-evidence fallback instead of failing the whole run |
| `GitHub did not return statistics for '...' after N attempts` | GitHub computes `/stats/*` endpoints asynchronously and was still warming the cache | Usually harmless (that repo's contribution is just skipped); raise `STATS_MAX_RETRIES`/`STATS_RETRY_DELAY_SECONDS` if it happens often |
| `INVALID: ... is missing required column(s)` | Dataset doesn't match the expected schema | Run `dataset_validator.py` (§5) and fix the CSV — this is what it's for |
| `Labelled dataset cache is missing row ...` | `predict_unlabeled_profiles.py` was pointed at a labeled dataset that hasn't been run through `train_on_real_dataset.py` yet | Run `train_on_real_dataset.py --dataset-path <same file>` first so the cache is populated |
| `Is_Fraud is not both classes present` | Labeled dataset validator caught an all-genuine or all-fraud file | Cross-validation needs both classes — check the file |
| `PermissionError` writing a dataset/results file | The CSV is open in Excel or another program | Close it and re-run |
| `MemoryError` mid-run | System memory pressure (unrelated to this pipeline — Python processes here are small) | Close other applications; none of these scripts hold more than a few thousand rows in memory |
| Model file overwritten unexpectedly | Ran `train_anomaly_model.py` without `--model-path`, which defaults to the production path | Pass `--model-path` explicitly for exploratory/smoke-test runs |

---

## 13. Known limitations (read before quoting numbers)

- **`train_anomaly_model.py` uses synthetic data.** It validates the
  pipeline end-to-end and justifies the model *architecture* choice
  (unsupervised IsolationForest vs. supervised alternatives), but is not an
  empirical finding about real candidates.
- **`train_on_real_dataset.py`'s n≈34 is small.** Every genuine row in the
  current labeled set has `mismatch_score = 0.00` (see the module docstring
  and `app/ml/README.md` §6) — there is no training coverage for a genuine
  candidate with a small, incidental discrepancy. This is why
  `predict_unlabeled_profiles.py`'s classifier over-flags real candidates
  (see its own output's summary block) — thresholding `mismatch_score`
  directly is more reliable than the trained classifier until the labeled
  set grows to cover that gap.
- **No dataset here should be presented as larger than it is.** If you scale
  the synthetic population up for a cleaner comparison chart, say so — it's
  a legitimate, real change to a data *generator*, not the same thing as a
  larger real dataset. See `synthetic_data.py`'s module docstring for the
  full caveat.
