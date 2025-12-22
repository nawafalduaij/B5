---
pagetitle: "W2 D5"
title: "Data Work (ETL + EDA)"
subtitle: "AI Professionals Bootcamp | Week 2"
date: 2025-12-25
---

## Policy: GenAI usage

- ✅ Allowed: **clarifying questions** (definitions, error explanations)
- ❌ Not allowed: generating code, writing solutions, or debugging by copy-paste
- If unsure: ask the instructor first

::: callout-tip
**In this course:** you build skill by typing, running, breaking, and fixing.
:::

# Day 5: Ship the Pipeline + Reporting Handoff

**Goal:** package your Week 2 work as a **job-ready handoff**: reproducible ETL → processed datasets → EDA notebook → exported figures → a short written summary with caveats.

::: {.muted}
Bootcamp • SDAIA Academy
:::

::: {.notes}
Say: “Today is about being employable: someone else can run your repo and trust your outputs.”
Do: show end-state: run_etl -> processed files -> notebook -> figures -> summary.md.
Ask: “If I clone your repo, can I reproduce your numbers in 5 minutes?”
:::

---

## Today’s Flow

* **Session 1 (60m):** ETL pipeline patterns (run_etl, config, logging, QA)
* *Asr Prayer (20m)*
* **Session 2 (60m):** Outputs + metadata + handoff quality (what to ship)
* *Maghrib Prayer (20m)*
* **Session 3 (60m):** Optional DuckDB “SQL view layer” + final checks
* *Isha Prayer (20m)*
* **Hands-on (120m):** Build `etl.py` + run metadata + `reports/summary.md` + final repo cleanup

---

## Learning Objectives

By the end of today, you can: 

* turn your work into a reproducible **ETL pipeline** (`run_etl()`)
* add lightweight logging and **fail-fast QA checks**
* write **idempotent** processed outputs (safe to rerun)
* generate a minimal **run metadata** JSON (inputs + counts)
* produce a short `reports/summary.md` with findings + caveats + next steps
* ensure a new person can clone and run your repo successfully

---

## Warm-up (5 minutes)

Confirm Day 3 + Day 4 artifacts exist.

**macOS/Linux**

```bash
source .venv/bin/activate
python scripts/run_day3_build_analytics.py
ls -la data/processed | head
```

**Windows PowerShell**

```powershell
.\\.venv\\Scripts\\Activate.ps1
python scripts\\run_day3_build_analytics.py
dir data\\processed
```

**Checkpoint:** `analytics_table.parquet` exists and Day 3 script runs.

---

## Definition of “done” for Week 2 (submit-ready)

You will ship: 

* **Reproducible ETL** that reads from `data/raw/` (and/or `data/cache/`)
* **Validations**: required columns, non-empty, unique keys, join validation, basic ranges
* **Idempotent outputs** to `data/processed/` (prefer Parquet)
* **EDA notebook** reading only from `data/processed/`
* **Exported figures** in `reports/figures/`
* **Written summary** in `reports/summary.md` (findings + caveats + next steps)

# Session 1

::: {.muted}
ETL pipeline patterns (run_etl, config, logging, QA)
:::

---

## Session 1 objectives

By the end of this session, you can: 

* define Extract / Transform / Load in this bootcamp context
* structure an ETL module with:

  * `load_inputs()`
  * `transform()`
  * `load_outputs()`
  * `run_etl()`
* add logging for row counts and paths
* make outputs idempotent (overwrite, don’t append)

---

## Context: notebooks are not pipelines

Notebooks are great for exploration.

Pipelines are needed when:

* you re-run many times
* you share work with teammates
* you want reproducibility and trust

Today: convert “daily scripts” into a clean ETL module. 

---

## Concept: ETL definition (Week 2)

ETL in our offline-first workflow: 

* **Extract:** read from `data/raw` and/or `data/cache`
* **Transform:** pure transforms (`df -> df`), deterministic
* **Load:** write to `data/processed` + minimal metadata

---

## A good ETL has these properties

* idempotent outputs (safe reruns)
* logs what it did (inputs/outputs/counts)
* checks assumptions (fail fast)
* keeps I/O separate from transforms
* produces stable, analysis-ready tables 

---

## Concept: minimal ETL module structure

In `src/bootcamp_data/etl.py`:

* `ETLConfig` dataclass
* `load_inputs(cfg)`
* `transform(orders, users, cfg)` *(or similar)*
* `load_outputs(df, cfg)`
* `write_run_meta(cfg, ...)`
* `run_etl(cfg)`

---

## Example: ETL skeleton (shape only)

```python
def load_inputs(cfg): ...
def transform(orders, users): ...
def load_outputs(df, cfg): ...
def run_etl(cfg): ...
```

::: {.muted}
We keep it small and shippable (no orchestration frameworks).
:::

---

## Concept: logging (why it matters)

Logging gives you:

* visibility (row counts)
* debugging info (paths, step names)
* audit trail (what ran, when)

Avoid “print-only pipelines” for anything beyond demos. 

---

## Example: minimal logging

```python
import logging
log = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log.info("Loading inputs...")
```

---

## Quick Check

**Question:** What 2 things should every ETL run log?

. . .

**Answer:** row counts and output paths (at minimum). 

---

## Concept: where QA checks go

Run checks:

* right after loading inputs
* right after major transforms (especially joins)
* before writing outputs

This catches problems closest to their cause. 

---

## Example: a QA checkpoint placement

```python
orders, users = load_inputs(cfg)
require_columns(orders, [...])
assert_unique_key(users, "user_id")

out = transform(orders, users)
assert len(out) == len(orders)  # join sanity

load_outputs(out, cfg)
```

---

## Micro-exercise: name 3 checks you will include (3 minutes)

Write 3 checks you will include in your `transform()`.

**Checkpoint:** each check has a clear purpose.

---

## Solution: examples of high ROI checks

* `require_columns(orders, ["order_id","user_id","amount","created_at","status"])`
* `assert_unique_key(users, "user_id")`
* `assert len(joined) == len(orders)` after left join

(Plus range checks for amount/quantity if needed.) 

---

## Session 1 recap

* ETL = Extract → Transform → Load (offline-first)
* Keep transforms pure and deterministic
* Log counts + paths
* Fail fast with lightweight QA checks
* Idempotent outputs are mandatory for reruns 

# Asr break {background-image='{{< brand logo anim >}}' background-opacity='0.1'}

## 20 minutes

**When you return:** we’ll define what “handoff quality” looks like and what to ship.

# Session 2

::: {.muted}
Outputs + metadata + handoff quality (what to ship)
:::

---

## Session 2 objectives

By the end of this session, you can: 

* define “handoff quality” for a data project
* produce minimal run metadata (config + counts)
* export a schema/missingness summary (optional)
* write a clear `reports/summary.md` with caveats

---

## Context: your future teammate is the user

A good repo answers:

* “How do I run this?”
* “Where are outputs?”
* “What do these columns mean?”
* “What can go wrong?”

This is what hiring managers look for too.

---

## Concept: handoff quality checklist

Your Week 2 handoff should include: 

* processed datasets (Parquet)
* EDA notebook (reads processed)
* exported figures
* written summary + caveats
* minimal metadata so numbers can be reproduced

---

## Concept: minimal run metadata

Run metadata can be a small JSON file with:

* config (input paths, output paths)
* row counts
* key quality stats (missing timestamps, match rate, etc.)

This makes reruns auditable. 

---

## Example: run metadata JSON

```json
{
  "rows_out": 12345,
  "inputs": {
    "orders_raw": "data/raw/orders.csv",
    "users_raw": "data/raw/users.csv"
  },
  "outputs": {
    "analytics_table": "data/processed/analytics_table.parquet"
  }
}
```

---

## Example: `write_run_meta(cfg, rows_out=...)` pattern 

```python
import json
from dataclasses import asdict

def write_run_meta(cfg, *, rows_out: int, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"config": {k: str(v) for k, v in asdict(cfg).items()}, "rows_out": rows_out}
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
```

---

## Micro-exercise: what else would you record? (3 minutes)

Add **two more fields** you would record in run metadata.

**Checkpoint:** two fields that would help debugging or reproducibility.

---

## Solution: helpful metadata fields

Examples:

* `missing_created_at_after_parse`
* `country_match_rate_after_join`
* `git_commit` (if you want, optional)
* `run_timestamp_utc`

Keep it minimal and useful. 

---

## Concept: written summary structure

Your `reports/summary.md` should include: 

* key findings (bulleted + quantified)
* definitions (metrics + filters)
* caveats (missingness, duplicates, join coverage, outliers)
* recommended next steps / questions

---

## Example: summary template (what to write)

```markdown
# Week 2 Summary

## Key findings
- ...
- ...

## Definitions
- Revenue = ...
- Refund rate = ...

## Data quality caveats
- Missingness: ...
- Joins: ...
- Outliers: ...

## Next questions
- ...
```

---

## Micro-exercise: write 3 caveats (5 minutes)

Write 3 caveats you must mention.

**Checkpoint:** each caveat is concrete and tied to your pipeline.

---

## Solution: common caveats (examples)

* Missingness in `amount` due to invalid parsing → totals may be understated.
* Some orders may not match users (join coverage < 100%) → segment results may be biased.
* Outliers exist in `amount`; charts used winsorized values for readability. 

---

## Session 2 recap

* Handoff quality = reproducible + understandable + auditable
* Add run metadata (config + counts + a few key stats)
* Ship a short summary with findings + caveats + next steps 

# Maghrib break {background-image='{{< brand logo anim >}}' background-opacity='0.1'}

## 20 minutes

**When you return:** optional DuckDB SQL + final checks before shipping.

# Session 3

::: {.muted}
Optional DuckDB “SQL view layer” + final checks
:::

---

## Session 3 objectives

By the end of this session, you can: 

* explain when SQL (DuckDB) is useful locally
* run a simple SQL aggregation on Parquet
* compare SQL output vs pandas output (sanity check)
* perform final repo checks before submission

---

## Context: sometimes SQL is the fastest expression

DuckDB lets you query local Parquet/CSV:

* no server
* offline
* quick analytics “view layer”

We treat this as optional — pandas remains the core. 

---

## Concept: DuckDB as a view layer

Use DuckDB when:

* the query is simpler in SQL than pandas
* you want a second way to verify results
* you want to practice SQL fundamentals without heavy setup 

---

## Example: query Parquet with DuckDB (minimal) 

```python
import duckdb
from pathlib import Path

def query_parquet(path: Path, sql: str):
    con = duckdb.connect()
    con.execute("CREATE VIEW t AS SELECT * FROM read_parquet(?)", [str(path)])
    return con.execute(sql).df()
```

---

## Micro-exercise: write a SQL query (4 minutes)

Write a SQL query for:

* revenue by country (sum of amount)
* sorted by revenue desc

**Checkpoint:** you wrote valid SQL text.

---

## Solution: SQL query example

```sql
SELECT
  country,
  COUNT(*) AS n,
  SUM(amount) AS revenue
FROM t
GROUP BY 1
ORDER BY revenue DESC
```

---

## Quick Check

**Question:** What is DuckDB reading from in this setup?

. . .

**Answer:** directly from the local Parquet file (no database server). 

---

## Final checks before shipping

Before you submit, confirm: 

* Day 5 ETL runs from a clean terminal
* outputs land in `data/processed/`
* notebook reads only processed data
* figures exported
* summary written
* README includes “How to run”

---

## Session 3 recap

* DuckDB is optional but useful for fast local SQL analytics
* It can help validate pandas results
* Final checklists catch “it works on my laptop” issues 

# Isha break {background-image='{{< brand logo anim >}}' background-opacity='0.1'}

## 20 minutes

**When you return:** we’ll implement `etl.py`, generate metadata, and write `reports/summary.md`.
