# Interview Audio Analysis & Self‑Reported Claim Extraction (CCS)

## Goal
Create a research‑grade, modular FastAPI service that accepts interview audio, transcribes it, cleans the transcript, extracts structured candidate claims via LLMs, stores the results, and prepares data for downstream fraud‑detection modules.

---

# 1. Full Architecture Design

```
[Client] --> FastAPI (REST) --> Queue (Redis/RabbitMQ) --> Worker(s)
                                            │                │
                                            │                └─► Whisper (audio → text)
                                            │
                                            └─► LLM Service (Gemini / GPT‑4o‑mini)

Database (Postgres/Mongo) stores:
    • Raw audio metadata
    • Transcripts & clean versions
    • Extracted claim JSON
    • Confidence scores & supporting sentences
    • Processing logs / error traces

Fraud‑Detection Service (separate repo) consumes claim JSON via
    • Direct DB access
    • Event stream (Kafka) / HTTP webhook
```

**Key components**
1. **API Layer** – FastAPI, Pydantic schemas, async endpoints, OpenAPI docs.
2. **Task Queue** – `celery` (Redis broker) or `RQ` for simplicity; enables async long‑running transcription & LLM calls.
3. **Audio Processor** – Wrapper around Whisper (fast‑whisper / WhisperX) exposing a Python callable.
4. **LLM Claim Extractor** – Service class that builds prompt, calls chosen LLM (Gemini‑2.5‑flash, GPT‑4o‑mini) with **strict JSON** schema via `response_schema`.
5. **Post‑processing** – filler‑word removal, tech‑skill normalization (spaCy + custom gazetteer), confidence aggregation.
6. **Persistence** – SQLAlchemy models for PostgreSQL *or* Motor ODM for MongoDB; repo pattern for testability.
7. **Explainability Layer** – Generates human‑readable rationale linked to supporting sentences.
8. **Monitoring** – `prometheus_fastapi_instrumentator`, structured logs, optional tracing (OpenTelemetry).

---

# 2. Folder Structure (Python project root: `interview_ccs/`)

```
interview_ccs/
├─ app/
│  ├─ api/                     # FastAPI routers
│  │   ├─ v1/                  # versioned routes
│  │   │   └─ audio.py         # /upload, /status endpoints
│  │   └─ deps.py               # common dependencies (db, queue)
│  ├─ core/                     # config, settings, logger
│  │   ├─ config.py
│  │   └─ logger.py
│  ├─ db/                       # ORM/ODM models & client
│  │   ├─ models.py
│  │   └─ client.py
│  ├─ services/                 # business logic
│  │   ├─ audio_processor.py   # Whisper wrapper
│  │   ├─ claim_extractor.py    # LLM interaction
│  │   ├─ postprocess.py       # cleaning, normalization
│  │   └─ explainer.py         # explainable output generator
│  ├─ tasks/                    # Celery/RQ workers
│  │   ├─ worker.py
│  │   └─ jobs.py              # transcription + extraction pipeline
│  ├─ schemas/                  # Pydantic request/response models
│  │   ├─ audio.py
│  │   └─ claim.py
│  └─ main.py                  # FastAPI app entry point
├─ tests/                       # unit & integration tests
│  ├─ api/
│  ├─ services/
│  └─ conftest.py
├─ scripts/                     # dev utilities (populate db, run demo)
│  └─ run_worker.sh
├─ Dockerfile
├─ docker-compose.yml           # db + redis + api + worker
├─ pyproject.toml               # poetry / pip dependencies
└─ README.md
```

---

# 3. Step‑by‑Step Implementation Plan

| Step | Description | Owner | Status |
|------|-------------|-------|--------|
| 1 | Initialize repository, add `pyproject.toml` with core deps (`fastapi`, `uvicorn`, `pydantic`, `sqlalchemy`/`motor`, `celery`, `redis`, `torch`, `faster-whisper`, `google‑generativeai`/`openai`, `spacy`, `sentence‑transformers`). | You | ❏ |
| 2 | Create `app/core/config.py` reading env vars (`HOST`, `PORT`, `DB_URL`, `REDIS_URL`, `LLM_PROVIDER`, `WHISPER_MODEL`). | You | ❏ |
| 3 | Implement DB client (`app/db/client.py`) with async SQLAlchemy engine (Postgres) *or* Motor client (Mongo). Provide `get_db` dependency. | You | ❏ |
| 4 | Define ORM/ODM models reflecting **AudioJob**, **Transcript**, **ClaimSet** (see Section 9). | You | ❏ |
| 5 | Build Pydantic schemas for request/response (`schemas/audio.py`, `schemas/claim.py`). Include `JobStatusEnum`. | You | ❏ |
| 6 | Write FastAPI router `api/v1/audio.py`:
   - `POST /v1/audio/upload` (multipart file, optional `candidate_id`).
   - Returns `job_id`.
   - `GET /v1/audio/status/{job_id}` returns processing state and final result when ready. | You | ❏ |
| 7 | Set up Celery app (`tasks/worker.py`) with Redis broker, result backend pointing to Postgres.
   - Register two tasks: `transcribe_audio` and `extract_claims`. | You | ❏ |
| 8 | Implement `AudioProcessor` (`services/audio_processor.py`):
   - Load Whisper model (faster‑whisper) once (singleton).
   - `async def transcribe(path: Path) -> str` returning raw transcript.
   - Support mp3, wav, m4a via `ffmpeg` (ensure it’s installed). | You | ❏ |
| 9 | Implement `PostProcessor` (`services/postprocess.py`):
   - Filler‑word list (`uh`, `um`, `you know`, etc.) removal using regex.
   - Sentence segmentation (spaCy `en_core_web_sm`).
   - Normalization of technology tokens via a gazetteer JSON (e.g., map "K8s" → "Kubernetes"). | You | ❏ |
|10| Implement `ClaimExtractor` (`services/claim_extractor.py`):
   - Build a **system prompt** describing the JSON schema (see Section 11).
   - Call LLM with `response_schema` for strict validation.
   - Parse and attach **confidence_score** (LLM may return `score` per field or we compute via token‑level log‑probability aggregation).
   - Return `ClaimSet` model. | You | ❏ |
|11| Implement `Explainer` (`services/explainer.py`):
   - Takes raw transcript and extracted claims.
   - Generates human‑readable explanations linking each claim to supporting sentences.
   - Output stored in DB for downstream fraud detection.
| You | ❏ |
|12| Wire the Celery pipeline in `tasks/jobs.py`:
   - `process_interview(job_id, file_path, candidate_id)` → calls `transcribe_audio` → `postprocess` → `extract_claims` → store results.
| You | ❏ |
|13| Add monitoring & error handling:
   - Global exception handlers for `AudioException`, `LLMException`.
   - Retry policies on Celery tasks (max 3 attempts, exponential back‑off).
| You | ❏ |
|14| Write unit tests for each service and integration test of the whole flow (mock Whisper and LLM calls). | You | ❏ |
|15| Containerize with Docker, add `docker‑compose.yml` (api, worker, redis, postgres). | You | ❏ |
|16| Draft research‑paper experiment plan (datasets, metrics – precision/recall of claim extraction, BLEU vs ground truth, confidence calibration, fraud‑detection downstream ROC‑AUC). | You | ❏ |

---

# 4. FastAPI Backend APIs

### 4.1 POST `/api/v1/audio/upload`
```json
Request (multipart/form-data):
    file: (binary)    # .mp3/.wav/.m4a
    candidate_id: str  # optional identifier linking to resume/GitHub record
Response (202 Accepted):
    {
        "job_id": "uuid",
        "status": "queued"
    }
```
* Returns a UUID that the client can poll.

### 4.2 GET `/api/v1/audio/status/{job_id}`
```json
Response (200):
    {
        "job_id": "uuid",
        "status": "completed" | "processing" | "failed",
        "result": {
            "transcript": "...",
            "claims": { ... },
            "explanations": ["..."],
            "confidence_score": 0.87
        }
    }
```
* When `status` == `completed`, `result` contains the final JSON.

---

# 5. Whisper Transcription Pipeline

1. **Model selection** – `faster-whisper` with `base` or `large-v2` depending on GPU availability.
2. **Audio pre‑processing** – Use `ffmpeg` to convert any supported format to 16 kHz mono PCM before feeding to model.
3. **Chunking for long files** – Split >30 min audio into 10‑minute windows, transcribe each, then concatenate with speaker‑agnostic timestamps.
4. **Output** – Raw string, optionally enriched with timestamps (useful for later evidence mapping).

---

# 6. Transcript Pre‑processing & Filler‑Word Removal

```python
FILLER_PATTERN = re.compile(r"\b(uh|um|you know|like|so|actually)\b", re.IGNORECASE)

def clean_transcript(text: str) -> str:
    # 1. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # 2. Remove filler words
    text = FILLER_PATTERN.sub("", text)
    # 3. Capitalize first letter of sentences (spaCy can help)
    return text
```
* The cleaned version is stored as `clean_transcript`.

---

# 7. Transcript Segmentation

* **Sentence segmentation** – spaCy `sentencizer` for speed.
* **Section detection** – simple rule‑based detection of typical interview phases (`"Experience:"`, `"Projects:"`, `"Leadership:"`).
* Each segment is passed to LLM with a *section‑aware* prompt (e.g., "Extract claims from the *Experience* section only.").

---

# 8. LLM Claim Extraction Pipeline

### 8.1 JSON Schema (Pydantic)
```python
class ClaimSet(BaseModel):
    technical_skills: List[str]
    frameworks: List[str]
    tools: List[str]
    programming_languages: List[str]
    cloud_platforms: List[str]
    databases: List[str]
    years_of_experience: Optional[str]
    job_roles: List[str]
    projects_claimed: List[str]
    leadership_claims: List[str]
    certifications: List[str]
    soft_skills: List[str]
    confidence_score: float
    supporting_sentences: List[str]
```

### 8.2 Prompt Template
```
You are an AI assistant that extracts self‑reported professional claims from a candidate's interview transcript. Return **ONLY** a JSON object that conforms to the schema below. Do not add explanations outside the JSON.
---
<SCHEMA>
---
Transcript:
"""
{segment_text}
"""
---
Extract all claims, fill arrays, and provide a **confidence_score** (0‑1) reflecting overall certainty. For each claim, include at least one supporting sentence in `supporting_sentences`.
If a claim is vague or uncertain, set `confidence_score` accordingly and optionally add "uncertain" notes inside the arrays.
```
* The schema is sent via Gemini/OpenAI `response_schema` to enforce strict validation.

### 8.3 Confidence Scoring Strategies
| Method | Description |
|--------|-------------|
| Log‑probability aggregation (model’s token‑level confidence) |
| Self‑reported `score` field from LLM (if model supports) |
| Heuristic based on **number of supporting sentences** / **presence of qualifiers** (e.g., "maybe", "I think") |

---

# 9. Database Schema (PostgreSQL example)

```sql
CREATE TABLE interview_job (
    id UUID PRIMARY KEY,
    candidate_id VARCHAR(64),
    original_filename VARCHAR(256),
    status VARCHAR(20) NOT NULL DEFAULT 'queued',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    finished_at TIMESTAMP WITH TIME ZONE,
    error TEXT
);

CREATE TABLE transcript (
    job_id UUID REFERENCES interview_job(id) ON DELETE CASCADE,
    raw TEXT NOT NULL,
    cleaned TEXT NOT NULL,
    language VARCHAR(10) DEFAULT 'en',
    PRIMARY KEY (job_id)
);

CREATE TABLE claim_set (
    job_id UUID PRIMARY KEY REFERENCES interview_job(id) ON DELETE CASCADE,
    json_data JSONB NOT NULL,      -- stores the ClaimSet object
    confidence FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
```
* MongoDB alternative: single collection with embedded sub‑documents.

---

# 10. Structured JSON Output Example
```json
{
  "technical_skills": ["Docker", "Kubernetes", "CI/CD"],
  "frameworks": ["FastAPI", "React"],
  "tools": ["Git", "Jenkins"],
  "programming_languages": ["Python", "Java"],
  "cloud_platforms": ["AWS", "GCP"],
  "databases": ["PostgreSQL", "MongoDB"],
  "years_of_experience": "5",
  "job_roles": ["Backend Engineer", "Team Lead"],
  "projects_claimed": ["Enterprise CRM migration", "Real‑time analytics platform"],
  "leadership_claims": ["Managed a team of 6 engineers"],
  "certifications": ["AWS Solutions Architect"],
  "soft_skills": ["Communication", "Problem solving"],
  "confidence_score": 0.92,
  "supporting_sentences": [
    "I have been working with Docker and Kubernetes for the past 4 years...",
    "I led a team of six engineers to deliver a cloud‑native CRM system..."
  ]
}
```

---

# 11. Validation Layer
* Use Pydantic model validation on the response from LLM. If validation fails, fall back to **re‑prompt** with "Please return a valid JSON according to the schema".
* Log schema violations for later analysis (helps evaluate LLM reliability).

---

# 12. Confidence Scoring & Uncertainty Handling
* After extraction, compute:
  - **field‑level confidence** (if LLM returns token‑level scores).
  - **overall confidence** = weighted average.
* Mark claims with qualifiers (`"maybe"`, `"probably"`) as low‑confidence and store a flag `is_uncertain` inside the DB for downstream fraud logic.

---

# 13. Error Handling & Retries
* **Audio errors** – unsupported codec → `AudioException` (return 422).
* **LLM errors** – rate‑limit, timeout → retry with exponential back‑off, max 3 attempts; on final failure, set job status `failed` and store error message.
* **Database errors** – transaction rollback, detailed log.

---

# 14. Async Processing & Queue Architecture
* Client receives `job_id` instantly; heavy work runs in Celery workers.
* Optional WebSocket endpoint (`/ws/status/{job_id}`) for real‑time progress updates.
* Task chaining: `transcribe >> postprocess >> extract_claims` using Celery canvas (`chain`).

---

# 15. Production‑Ready Modular Code Tips
* **Dependency injection** – FastAPI `Depends` for DB session, LLM client, audio processor.
* **Configuration management** – `pydantic.BaseSettings` reading from `.env`.
* **Logging** – JSON‑structured logs (`structlog`) with correlation IDs (job_id).
* **Testing** – Mock Whisper (`unittest.mock`) and LLM (`responses` library) to keep CI fast.
* **Docker** – Multi‑stage build: builder stage installs torch + whisper, final stage contains only runtime deps.

---

# 16. Research‑Grade Enhancements & Novelty
| Feature | Why it adds research value |
|---------|---------------------------|
| **Section‑aware prompting** | Allows experiment on how context windows affect claim recall. |
| **Embedding similarity sanity check** | Compute cosine similarity between extracted skill tokens and transcript sentences using `sentence‑transformers`; report a similarity score to gauge extraction faithfulness. |
| **Cross‑modal verification** | Align claim timestamps with audio phoneme confidence (whisper `word_timestamps`) to detect fabricated claims spoken with hesitation. |
| **Multi‑model fallback** | If Gemini fails, automatically fallback to OpenAI GPT‑4o‑mini; log model choice for ablation study. |
| **Explainable AI** | Store per‑claim rationale (supporting sentence + similarity) – useful for human evaluation and for the paper’s “interpretability” section. |
| **Temporal extraction** | Regex + LLM to pull explicit time spans ("for 3 years", "since 2018") enabling timeline consistency checks across resume and interview. |
| **Anomaly detection** | Train a lightweight Isolation Forest on claim vectors to flag outliers before downstream fraud module. |

---

# 17. Deployment Strategy
* **Local dev** – `docker-compose up -d` (api, worker, redis, postgres).
* **Cloud** – Deploy API & worker as separate services on Kubernetes; use CloudSQL for Postgres and Cloud Redis.
* **CI/CD** – GitHub Actions: lint → unit tests → build Docker image → push to registry → Helm upgrade.

---

# 18. Evaluation Metrics & Experimentation Methodology
| Metric | Definition |
|--------|------------|
| **Extraction Precision/Recall** | Compare extracted claim lists against a manually annotated ground‑truth set (≈200 interview samples). |
| **Schema Validity Rate** | % of LLM responses that pass Pydantic validation on first try. |
| **Confidence Calibration** | Use Expected Calibration Error (ECE) to see if confidence_score aligns with actual correctness. |
| **Processing Latency** | Avg time from upload to final JSON (target < 30 s for 10‑min audio). |
| **Resource Utilization** | GPU memory consumption of Whisper model, average CPU for LLM calls. |
| **Fraud‑Detection downstream ROC‑AUC** | End‑to‑end pipeline performance when feeding claims into the fraud module (baseline vs enhanced claims). |

---

# 19. Diagram Suggestions for Paper
1. **System Architecture Diagram** – showing API ↔ Queue ↔ Workers ↔ Whisper ↔ LLM ↔ DB.
2. **Data Flow Chart** – audio → transcript → cleaning → segmentation → claim extraction → explanation.
3. **Model fallback diagram** – Gemini → OpenAI fallback.
4. **Evaluation pipeline** – ground‑truth annotation → extraction → metrics.

---

# 20. Next Steps
* Review this implementation plan.
* Confirm preferred database (Postgres vs Mongo) and queue library (Celery vs RQ).
* Choose Whisper model size and LLM provider.
* Approve any additional research experiments you’d like to include.

---

*Please provide feedback on the plan, any constraints, or preferences so we can progress to concrete code implementation.*
