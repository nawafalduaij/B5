---
pagetitle: "W2 D3"
title: "Data Work (ETL + EDA)"
subtitle: "AI Professionals Bootcamp | Week 2"
date: 2025-12-23
---

## Policy: GenAI usage

- ✅ Allowed: **clarifying questions** (definitions, error explanations)
- ❌ Not allowed: generating code, writing solutions, or debugging by copy-paste
- If unsure: ask the instructor first

::: callout-tip
**In this course:** you build skill by typing, running, breaking, and fixing.
:::

# Day 3: Datetimes + Outliers + Safe Joins

**Goal:** build an **analytics-ready table** by parsing time correctly, handling outliers safely, and joining without corrupting your data. 

::: {.muted}
Bootcamp • SDAIA Academy
:::

::: {.notes}
Say: “Today is where most real-world analytics bugs happen: time + joins.”
Do: show a left join that explodes, then show validate= preventing it.
Ask: “If your join doubled rows, would you notice?”
:::

---

## Today’s Flow

* **Session 1 (60m):** Datetime parsing + time parts
* *Asr Prayer (20m)*
* **Session 2 (60m):** Outliers + sanity checks + core pandas ops
* *Maghrib Prayer (20m)*
* **Session 3 (60m):** Safe joins + join validation + `.pipe()` design
* *Isha Prayer (20m)*
* **Hands-on (120m):** Build `analytics_table.parquet` (time + join + outlier flags)

---

## Learning Objectives

By the end of today, you can: 

* parse datetime columns safely using `pd.to_datetime(..., errors="coerce")`
* add time parts (date/month/day-of-week/hour) for grouping
* identify outliers using percentiles / IQR and choose a safe policy
* use core pandas ops for real work (`.loc`, `.assign`, `groupby/agg`)
* join tables safely with `merge(validate=...)` and detect join explosions
* write `data/processed/analytics_table.parquet`

---

## Warm-up (5 minutes)

Run Day 2 and confirm cleaned output exists.

**macOS/Linux**

```bash
source .venv/bin/activate
python scripts/run_day2_clean.py
python -c "import pandas as pd; df=pd.read_parquet('data/processed/orders_clean.parquet'); print(len(df)); print(df.dtypes)"
```

**Windows PowerShell**

```powershell
.\\.venv\\Scripts\\Activate.ps1
python scripts\\run_day2_clean.py
python -c "import pandas as pd; df=pd.read_parquet('data/processed/orders_clean.parquet'); print(len(df)); print(df.dtypes)"
```

**Checkpoint:** `orders_clean.parquet` exists and loads without errors.

---

## Common setup issue: `ModuleNotFoundError: bootcamp_data`

If you see:

```text
ModuleNotFoundError: No module named 'bootcamp_data'
```

it usually means Python can’t see your **`src/`** folder.

**Fix (recommended, once per repo):** from repo root:

```bash
pip install -e .
```

**Fix (quick + works for scripts):** add this to the top of `scripts/*.py` **before** importing `bootcamp_data`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
```

---

## Inputs we assume exist (from Day 1–2)

Today’s script expects these files:

* `data/processed/orders_clean.parquet` *(produced Day 2)*
* `data/processed/users.parquet` *(produced Day 1)*

Quick check:

```bash
python -c "from pathlib import Path; p=Path('data/processed'); print([x.name for x in p.glob('*.parquet')])"
```

If either file is missing, rerun:

* `python scripts/run_day1_load.py`
* `python scripts/run_day2_clean.py`

---

## Where we are in the weekly workflow

Canonical flow: 

1. Load ✅
2. Verify ✅
3. Clean ✅
4. Transform ✅ *(today starts)*
5. Analyze (Day 4)
6. Visualize (Day 4)
7. Conclude (Day 5)

# Session 1

::: {.muted}
Datetime parsing + time parts
:::

---

## Session 1 objectives

By the end of this session, you can: 

* explain why datetime parsing is a correctness problem
* parse timestamps safely with `errors="coerce"`
* handle time zones intentionally (naive vs UTC-aware)
* add time parts for grouping (month, day-of-week, hour)

---

## Context: time-based analysis is fragile

If time is wrong, you get wrong:

* trends
* cohorts
* “before vs after”
* “most recent” sorting

Datetime mistakes often don’t crash — they silently lie. 

---

## Concept: timestamps vs strings

A timestamp should be:

* a real datetime type (`datetime64[ns]`, often UTC-aware)
* sortable by time
* groupable by month/week/day

A string “looks like time” but behaves like text.

---

## Quick demo: sorting strings can mislead

If you sort strings:

* `"2025-2-1"` may come after `"2025-12-1"` (lexicographic)

If you sort datetimes:

* you get true chronological order

---

## Concept: safe parsing default

Use:

* `pd.to_datetime(..., errors="coerce", utc=True)` 

Why?

* invalid formats become missing (you can measure/fix)
* `utc=True` prevents mixing naive and aware timestamps

---

## Example: `parse_datetime(df, col)`

```python
import pandas as pd

def parse_datetime(
    df: pd.DataFrame,
    col: str,
    *,
    utc: bool = True,
) -> pd.DataFrame:
    dt = pd.to_datetime(df[col], errors="coerce", utc=utc)
    return df.assign(**{col: dt})
```
---

## Micro-exercise: parse `created_at` (6 minutes)

1. Load `data/processed/orders_clean.parquet`
2. Parse `created_at` using `errors="coerce"` and `utc=True`
3. Count missing timestamps after parsing

**Checkpoint:** `created_at` dtype becomes datetime and you can count missing values.

---

## Solution (example) {.smaller}

```python
import pandas as pd

orders = pd.read_parquet("data/processed/orders_clean.parquet")
orders2 = parse_datetime(orders, "created_at", utc=True)

print(orders2["created_at"].dtype)
print("n_missing_created_at:", orders2["created_at"].isna().sum())
```

---

## Quick Check

**Question:** What does `errors="coerce"` do?

. . .

**Answer:** invalid datetime strings become missing values instead of raising an exception. 

---

## Pitfall: ambiguous date formats

`03/04/2025` could be:

* March 4 (MM/DD)
* April 3 (DD/MM)

Rule: don’t guess. If you have ambiguous formats, you need a plan. 

---

## Concept: time zones (naive vs aware)

* **Naive**: no timezone info (dangerous when mixing sources)
* **Aware**: has timezone (e.g., UTC)

Operational default:

* store event timestamps in **UTC**
* convert to local time only for display (later)

---

## Quick Check

**Question:** If some timestamps are UTC and some are local time, what breaks?

. . .

**Answer:** comparisons and ordering (you might put events in the wrong day/hour).

---

## Concept: add time parts for grouping

Common derived fields: 

* `date` (for daily tables)
* `month` (for trend grouping)
* `dow` (day-of-week)
* `hour` (hour-of-day)

These turn “timestamp” into “group keys”.

---

## Example: `add_time_parts(df, ts_col)`

```python
import pandas as pd

def add_time_parts(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    ts = df[ts_col]
    return df.assign(
        date=ts.dt.date,
        year=ts.dt.year,
        month=ts.dt.to_period("M").astype("string"),
        dow=ts.dt.day_name(),
        hour=ts.dt.hour,
    )
```
---

## Micro-exercise: add month + day-of-week (6 minutes)

1. Parse `created_at`
2. Add time parts
3. Print the first 5 rows of: `created_at`, `month`, `dow`, `hour`

**Checkpoint:** these columns exist and look reasonable.

---

## Solution (example) {.smaller}

```python
orders3 = (
    orders.pipe(parse_datetime, col="created_at", utc=True)
          .pipe(add_time_parts, ts_col="created_at")
)

print(orders3[["created_at","month","dow","hour"]].head())
```

---

## Why we used `.pipe()` here

`.pipe()` helps you read transformations top-to-bottom:

* load
* parse
* add time parts

This is the “pure functions” style we’ll use all week. 

---

## Micro-exercise: one question you can now answer (3 minutes)

Write ONE question you can answer now that you have `month` and `dow`.

Examples:

* “How many orders per month?”
* “Do refunds happen more on certain days?”

**Checkpoint:** a single, measurable sentence.

---

## Solution: good time-based questions

Examples:

* “How many orders per month (n)?”
* “What is refund rate by day-of-week?”
* “What hour has the most orders?”

The key: quantify and compare. 

---

## Quick Check

**Question:** Why do we add time parts instead of grouping by raw timestamps?

. . .

**Answer:** raw timestamps are too granular; time parts create useful group keys (month/day/hour).

---

## Session 1 recap

* Parse timestamps safely (`errors="coerce"`, often `utc=True`)
* Measure invalid timestamps (missing after parse)
* Add time parts to make grouping easy (month/dow/hour)
* Prefer pipelines of pure transforms with `.pipe()` 

# Asr break {background-image='{{< brand logo anim >}}' background-opacity='0.1'}

## 20 minutes

**When you return:** we’ll handle outliers and practice pandas ops that matter daily.

# Session 2

::: {.muted}
Outliers + sanity checks + core pandas ops
:::

---

## Session 2 objectives

By the end of this session, you can: 

* explain why outliers distort averages and charts
* compute percentile summaries (p1/p50/p99)
* compute IQR bounds and flag outliers
* choose a safe outlier policy (flag / cap for visualization)
* use core pandas operations:

  * selection (`.loc`)
  * assignment (`.assign`)
  * groupby/agg for totals and rates

---

## Context: outliers are not always “bad”

Outliers could be: 

* data entry errors (wrong units)
* genuine rare events (VIP purchases)
* fraud spikes
* price changes

Rule: don’t delete first. Identify + decide.

---

## Concept: why outliers distort means

Mean is sensitive to extremes.

If one order is 1000× bigger than others:

* mean revenue jumps
* charts flatten
* “typical order” becomes misleading

Better: include median and percentiles. 

---

## Quick Check

**Question:** Which is more robust to outliers: mean or median?

. . .

**Answer:** median.

---

## Concept: start with percentiles

A simple first look:

* p50 (median)
* p90
* p99

If p99 is wildly larger than p50, you likely have a heavy tail or errors. 

---

## Percentiles cheat sheet (p1 / p50 / p99)

* **p1**: value below which **1%** of observations fall (bottom 1%)
* **p50**: median (“typical”)
* **p99**: value below which **99%** of observations fall (top 1% are above)

::: {.muted}
Winsorization often caps to **p1/p99** to make charts readable without deleting rows.
:::

---

## Example: percentile summary (tiny)

```python
s = orders["amount"].dropna()
print(s.quantile([0.5, 0.9, 0.99]))
```

---

## Micro-exercise: compute p50/p90/p99 (6 minutes)

1. Load `orders_clean.parquet`
2. Compute p50, p90, p99 for `amount` (ignore missing)
3. Write down the numbers

**Checkpoint:** you can explain what each percentile means.

---

## Solution (example)

```python
import pandas as pd
orders = pd.read_parquet("data/processed/orders_clean.parquet")
s = orders["amount"].dropna()
print(s.quantile([0.5, 0.9, 0.99]))
```

---

## Concept: IQR bounds (practical)

IQR method: 

* Q1 = 25th percentile
* Q3 = 75th percentile
* IQR = Q3 - Q1
* bounds: `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`

We use bounds to flag “extreme” values.

---

## Example: `iqr_bounds(s)` 

```python
def iqr_bounds(s, k=1.5):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - k*iqr), float(q3 + k*iqr)
```

---

## Micro-exercise: count outliers (7 minutes)

1. Compute IQR bounds for `amount`
2. Count how many rows are outside bounds

**Checkpoint:** you can print `n_outliers`.

---

## Solution (example) {.smaller}

```python
import pandas as pd

orders = pd.read_parquet("data/processed/orders_clean.parquet")
s = orders["amount"].dropna()

lo, hi = iqr_bounds(s, k=1.5)
n_out = ((orders["amount"] < lo) | (orders["amount"] > hi)).sum()
print("bounds:", lo, hi)
print("n_outliers:", int(n_out))
```

---

## Concept: safe outlier policy for EDA

Good defaults for Week 2 analytics work: 

* keep rows (don’t delete silently)
* add an `is_outlier` flag
* optionally create a capped version for charts (`winsorize`)

---

## Example: winsorize (cap for viz) 

```python
def winsorize(s, lo=0.01, hi=0.99):
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(lower=a, upper=b)
```

---

## Micro-exercise: create `amount_winsor` (6 minutes)

Create a new column:

* `amount_winsor = winsorize(amount, 0.01, 0.99)`

**Checkpoint:** min/max of `amount_winsor` are within p1/p99.

---

## Solution (example)

```python
orders2 = orders.assign(amount_winsor=winsorize(orders["amount"]))
print(orders2["amount_winsor"].min(), orders2["amount_winsor"].max())
```

---

## Quick Check

**Question:** Why do we cap values for visualization instead of deleting outliers?

. . .

**Answer:** deletion can remove real events; capping preserves rows while making charts readable. 

---

## Transition: pandas operations you’ll use daily

Most data work is:

* select rows/columns
* assign new columns
* group and aggregate
* reshape occasionally

You don’t need 100 pandas tricks — you need a few high-ROI ones. 

---

## Core pattern 1: selection with `.loc`

::: callout-tip
Avoid chained indexing. Use `.loc[...]` for clarity and correctness. 
:::

Example:

```python
paid = orders.loc[orders["status_clean"].eq("paid"), ["user_id","amount"]]
```

---

## Micro-exercise: filter “paid” orders (5 minutes)

1. Filter rows where `status_clean == "paid"`
2. Keep columns: `user_id`, `amount`
3. Count rows

**Checkpoint:** you can print `n_paid`.

---

## Solution (example)

```python
paid = orders.loc[orders["status_clean"].eq("paid"), ["user_id","amount"]]
print("n_paid:", len(paid))
```

---

## Core pattern 2: assignment with `.assign`

Use `.assign()` to add columns without messing with chained assignment. 

Example:

```python
orders2 = orders.assign(amount_usd=lambda d: d["amount"] * 1.0)
```

---

## Micro-exercise: add a boolean column (4 minutes)

Add:

* `is_refund = (status_clean == "refund")`

**Checkpoint:** `is_refund` exists and contains True/False.

---

## Solution (example)

```python
orders2 = orders.assign(is_refund=orders["status_clean"].eq("refund"))
print(orders2["is_refund"].value_counts(dropna=False))
```

---

## Core pattern 3: groupby/agg

Most analytics tables come from:

* `groupby`
* `agg`
* `reset_index`

Example pattern: 

```python
summary = (
  orders.groupby("user_id")
        .agg(n=("order_id","size"), revenue=("amount","sum"))
        .reset_index()
)
```

---

## Micro-exercise: revenue per user (7 minutes)

Compute per-user summary:

* `n_orders`
* `revenue` (sum of amount)

**Checkpoint:** you produce a DataFrame with one row per user.

---

## Solution (example) {.smaller}

```python
per_user = (
    orders.groupby("user_id", dropna=False)
          .agg(
              n_orders=("order_id", "size"),
              revenue=("amount", "sum"),
              aov=("amount", "mean"),
              med_amount=("amount", "median"),
          )
          .reset_index()
)
print(per_user)
```

---

## Concept: rates need numerator + denominator

“Refund rate” example: 

* numerator: number of refunds
* denominator: total orders

Always show both, then compute the ratio.

---

## Example: refund rate table (pattern) 

```python
rate = (
    orders.assign(is_refund=orders["status_clean"].eq("refund"))
          .groupby("user_id")
          .agg(refunds=("is_refund","sum"), total=("is_refund","size"))
          .assign(refund_rate=lambda t: t["refunds"] / t["total"])
          .reset_index()
)
```

---

## Quick Check

**Question:** Why should you include sample size (`total`) with a rate?

. . .

**Answer:** because rates from small groups are noisy and misleading. 

---

## Optional: tidy data mental model (preview)

Tidy idea: 

* each variable is a column
* each observation is a row
* each observational unit is a table

We reshape when data is not tidy.

---

## Example: wide → long with `melt` 

```python
long = df.melt(
    id_vars=["user_id","date"],
    value_vars=["clicks","views"],
    var_name="metric",
    value_name="value",
)
```

---

## Micro-exercise: predict the output (3 minutes)

If `metric` is `"clicks"` and `"views"`:

* how many rows will `long` have compared to the original?

**Checkpoint:** answer in one sentence.

---

## Solution: predict the output

If you melt 2 value columns, you usually get about **2× as many rows** (one row per metric per original row).

---

## Preview: long → wide with `pivot_table` (optional)

`melt` goes **wide → long**.  
`pivot_table` goes **long → wide**.

```python
wide = long.pivot_table(
    index=["user_id", "date"],
    columns="metric",
    values="value",
    aggfunc="sum",  # ⚠ if keys aren't unique, pivot will aggregate
).reset_index()
```

::: callout-tip
If you expected *no aggregation*, but pivot needs `aggfunc`, you probably don’t have unique keys.
:::

---

## Session 2 recap

* Start with percentiles and IQR to understand outliers
* Prefer: flag outliers + cap for visualization (don’t delete silently) 
* Use high-ROI pandas ops:

  * `.loc`, `.assign`, `groupby/agg`
* Rates require numerator + denominator

# Maghrib break {background-image='{{< brand logo anim >}}' background-opacity='0.1'}

## 20 minutes

**When you return:** we’ll join orders + users safely and prevent join explosions.

# Session 3

::: {.muted}
Safe joins + join validation + `.pipe()` design
:::

---

## Session 3 objectives

By the end of this session, you can: 

* explain common join types (left / inner / outer) in operational terms
* prevent join explosions with key checks + `validate=...`
* detect key dtype mismatches (`"001"` vs `1`)
* implement a `safe_left_join(...)` helper
* build an “analytics table” from orders + users

---

## Context: joins are the #1 analytics disaster

Most wrong dashboards come from wrong joins:

* duplicated rows (join explosion)
* missing matches (dtype mismatch)
* silently dropped data (inner join used by accident)

We prevent this with pre-checks + validation. 

---

## Concept: join types (operational meaning)

* **Left join**: keep all rows from main table, enrich with lookup (most common) 
* **Inner join**: keep only matches (can silently drop data)
* **Outer join**: reconciliation (see what’s missing on each side)

---

## Quick Check

**Question:** If you want to keep all orders, which join is most appropriate?

. . .

**Answer:** left join (orders left, users right).

---

## Concept: join cardinality (why validation exists)

Common situations:

* orders (many) → users (one)
* events (many) → lookup table (one)

If “one side” isn’t unique, your left join can multiply rows.

That’s a join explosion. 

---

## Example: the row count rule of thumb

For a true left join:

* `len(joined) == len(left)` should hold

If rows increased:

* you probably joined on a non-unique key (many-to-many)

---

## Micro-exercise: join sanity check (3 minutes)

If `orders` has 10 rows and `users` has 4 rows…

After a left join of `orders` onto `users`, how many rows should you have?

**Checkpoint:** answer with a number (and why).

---

## Solution: join sanity check

You should still have **10 rows** (all orders remain).
If you got >10, you likely had a join explosion.

---

## Pitfall: dtype mismatch breaks matches

If `orders.user_id` is `"0007"` (string) but `users.user_id` is `7` (int):

* join matches fail
* enriched columns become missing (`NaN`)

This is why Day 1 forced IDs as strings. 

---

## Micro-exercise: spot the mismatch (4 minutes)

Which join keys will match?

A) left has `"001"`, right has `"001"`
B) left has `"001"`, right has `1`
C) left has `1`, right has `1`

**Checkpoint:** choose A/B/C that will match.

---

## Solution: spot the mismatch

* A matches (string == string)
* B does **not** match (string != int)
* C matches (int == int)

---

## Concept: validate joins with pandas

Pandas `merge` supports `validate=` to enforce expected cardinality. 

Examples:

* `"many_to_one"`: left may repeat keys; right must be unique
* `"one_to_one"`: both sides unique
* `"many_to_many"`: allowed, but dangerous for analytics tables

---

## Example: safe left join helper

```python
import pandas as pd

def safe_left_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | list[str],
    *,
    validate: str,
    suffixes: tuple[str, str] = ("", "_r"),
) -> pd.DataFrame:
    return left.merge(
        right,
        how="left",
        on=on,
        validate=validate,
        suffixes=suffixes,
    )
```

---

## Micro-exercise: choose validate option (3 minutes)

We join:

* orders (many rows per user)
* users (one row per user)

Which validate is correct?

A) `one_to_one`
B) `many_to_one`
C) `many_to_many`

**Checkpoint:** choose A/B/C.

---

## Solution: choose validate option

**B) `many_to_one`** (many orders map to one user). 

---

## Concept: pre-check the “one” side

Before joining, assert the lookup table is unique: 

* `assert_unique_key(users, "user_id")`

This catches join explosions early.

---

## Debug drill: reproduce a join explosion (predict)

Imagine `users` accidentally has two rows for `user_id="0001"`.

What happens to orders for `"0001"` after a left join?

. . .

Answer: each order for `"0001"` duplicates (row count increases).

---

## Example: post-join checks (row count + match rate)

After join:

* assert row count didn’t change
* measure match rate in new columns (e.g., `country` missing rate)

These are cheap and powerful. 

---

## Micro-exercise: join and check row count (7 minutes)

1. Load `orders_clean.parquet` and `users.parquet`
2. Run `assert_unique_key(users, "user_id")`
3. Join with `safe_left_join(..., validate="many_to_one")`
4. Assert row count stayed the same

**Checkpoint:** join succeeds and row counts match.

---

## Solution (example) {.smaller}

```python
import pandas as pd
from bootcamp_data.quality import assert_unique_key
from bootcamp_data.transforms import parse_datetime, add_time_parts
from bootcamp_data.joins import safe_left_join  # (we'll create this in hands-on)

orders = pd.read_parquet("data/processed/orders_clean.parquet")
users  = pd.read_parquet("data/processed/users.parquet")

assert_unique_key(users, "user_id")

joined = safe_left_join(orders, users, on="user_id", validate="many_to_one", suffixes=("", "_user"))
assert len(joined) == len(orders), "Row count changed (possible join explosion)"
print(joined[["user_id","country"]].head())
```

::: {.notes}
If bootcamp_data.joins does not exist yet, they will implement it in hands-on.
:::

---

## Quick Check

**Question:** What does `validate="many_to_one"` protect you from?

. . .

**Answer:** it prevents “users” from having duplicate keys that would multiply rows in the join. 

---

## Concept: transformation design (pure functions + piping)

Good ETL design: 

* transforms are **pure**: `df -> df`
* I/O is separate from transforms
* pipelines read top-to-bottom via `.pipe()`

This makes your ETL readable and testable.

---

## Example: analytics table builder (pattern)

```python
def build_analytics_table(orders, users):
    return (
        orders
        .pipe(parse_datetime, col="created_at", utc=True)
        .pipe(add_time_parts, ts_col="created_at")
        .pipe(
            lambda d: safe_left_join(
                d,
                users,
                on="user_id",
                validate="many_to_one",
                suffixes=("", "_user"),
            )
        )
    )
```

---

## Micro-exercise: explain the pipeline (3 minutes)

In one sentence:

What does each `.pipe(...)` step do?

**Checkpoint:** you can narrate the pipeline without looking at code details.

---

## Solution: explain the pipeline

* parse time → adds datetime type
* add time parts → creates grouping keys
* safe left join → enrich orders with user fields without row explosions

---

## Session 3 recap

* Left join is the default for analytics enrichment
* Validate cardinality with `merge(validate=...)` 
* Pre-check uniqueness on the “one” side
* Post-check row count + match rate
* Use pure transforms + `.pipe()` to build an analytics-ready table

# Isha break {background-image='{{< brand logo anim >}}' background-opacity='0.1'}

## 20 minutes

**When you return:** we’ll implement datetime + outlier + join helpers and write `analytics_table.parquet`.
