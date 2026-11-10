# Rural Triage · Backend

[![CI](https://github.com/UbeimarY/triage-rural-backend/actions/workflows/ci.yml/badge.svg)](https://github.com/UbeimarY/triage-rural-backend/actions/workflows/ci.yml)

API for the offline-first rural triage PWA
([`triage-rural`](https://github.com/UbeimarY/triage-rural)). Phones work without signal and upload
their outbox here when they reconnect; coordinators follow every synchronized visit.

> Academic prototype with **fictitious data only**. It supports prioritization; it does not diagnose.

## Endpoints

| Method | Path | Who | Purpose |
|---|---|---|---|
| POST | `/auth/login` | everyone | Access token (7 days: promoters spend days without signal) |
| GET | `/auth/me` | signed in | Current account |
| POST | `/sync/push` | promoters | Apply a batch of outbox operations, one answer per operation |
| GET | `/dashboard/summary` | coordinators | Counts by priority and community, pending reviews |
| GET | `/dashboard/visits` | coordinators | Synchronized visits, most urgent first |
| GET | `/status` · `/healthz` | everyone | Health checks |

Interactive documentation: `/docs` (OpenAPI).

## Synchronization design

- **Idempotent:** every outbox operation carries a client UUID. An operation already applied is
  answered `duplicate`, never applied twice, so retries on bad networks are safe.
- **Optimistic concurrency, no silent overwrites:** an edit states the version it was based on.
  If the server has a newer one, the answer is `conflict` with the server's copy, and a person
  decides on the phone.
- **Isolation:** each operation runs in its own savepoint; one invalid record never blocks the
  batch. Possible answers: `applied`, `duplicate`, `retry` (a dependency has not arrived yet),
  `conflict`, `rejected` (never accepted, for example without consent).
- **Ownership:** promoters can only write their own patients and visits.
- **Row locks** (`SELECT … FOR UPDATE`) when two devices edit the same visit at the same time.

## AI training (`ml/`)

The text classifier runs **in the browser**; this repository trains and exports it.

```bash
uv run python -m ml.train   # writes ml/artifacts/
```

- `ml/dataset.py`: synthetic, documented dataset with scenario families; train/test split **by
  scenario**, so test descriptions are combinations never seen in training.
- `ml/features.py`: the text features (version 2), reproduced in TypeScript by the app:
  - **Denial:** "sin", "niega", "nunca", "ni" and a "no" before a verb of presence
    ("no tiene", "no presenta", "no se ha…") mark the next content words as denied, until
    the clause ends. In "no puede hablar" the "no" describes the symptom and is kept.
  - **Typos:** four-letter pieces of every affirmed word, so "fievre" still looks like "fiebre".
  - **Positions:** every word keeps where it is in the text, so the app highlights it.
- `ml/train.py`: TF-IDF + logistic regression + **temperature scaling**; decision threshold
  chosen by grouped cross-validation for **recall(high) ≥ 0.95**; v2 is compared with the v1
  method **trained on the same data**.
- `ml/artifacts/`: exported model (copied to the Angular app), **model card** with metrics and
  limitations, and equivalence fixtures the TypeScript tests must reproduce.

| Same data, held-out | v1: words + word pairs | **v2** |
|---|---|---|
| Recall of urgent cases, new scenarios | 0.986 | **1.000** |
| Recall of urgent cases, with typos | 0.955 | **1.000** |
| Denied urgent findings wrongly called urgent | 41.4 % | **5.7 %** |
| Expected calibration error | 0.060 | **0.004** |

The data is synthetic, so these numbers do not predict real-world performance; denial is
handled by rules, not understanding. See the model card for the limitations.

## Privacy (Ley 1581 de 2012)

Minimal patient data (code, birth year, sex, community), consent required and enforced by a
database constraint, adults only, no public sign-up, role-based access.

## Run locally

```bash
docker compose up -d                       # PostgreSQL on port 5433
cp .env.example .env                       # then set JWT_SECRET
uv sync
uv run alembic upgrade head
TRIAGE_PASSWORD='…' uv run python -m triage.cli create-user --email ana@example.com --name "Ana" --role promoter
uv run uvicorn triage.app:create_app --factory --reload --port 8001
uv run pytest
```

## Deployment

Docker image (`Dockerfile`, two stages, prebuilt wheels only, non-root user) on Fly.io
(`fly.toml`), PostgreSQL on Neon. Secrets (`DATABASE_URL`, `JWT_SECRET`) are set with
`fly secrets set`, never committed.

## Tech stack

FastAPI · SQLAlchemy 2 (async) · asyncpg · Alembic · PostgreSQL · Argon2 · JWT · scikit-learn
(training only) · uv · Ruff · pytest · GitHub Actions.
