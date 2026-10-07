# CLAUDE.md

## Project

**fraud-detection-databricks**: a rebuild of my fraud detection pipeline on Databricks Free Edition using PySpark, Delta Lake and MLflow, trained on the IEEE-CIS Fraud Detection dataset (Kaggle).

Why it exists: to back up "Spark / Databricks" on my CV for applied/fraud Data Scientist roles (the target is Signifyd, Data Scientist II, Applied Decision Science). It follows on from my earlier repo `github.com/liamhavers/fraud-detection-system`, which covers credit card fraud with XGBoost, FastAPI, Docker and PSI drift monitoring. This repo is the "built for scale" version of that project.

Full step-by-step build guide (with reference code): https://claude.ai/code/artifact/a3518b2e-5166-4fb2-a8ef-35d4d0e23314

## How I want you to work with me

- I have to explain every design decision in interviews **without AI help**. When you write or change Spark code, add a short explanation of the key concept involved (shuffles, window frames, broadcast joins, skew, lazy evaluation) and why we chose this approach.
- Work one step at a time, following the Build plan below. Don't jump ahead or scaffold later steps unless I ask.
- Prefer small, reviewable changes. Show me the diff before large refactors.
- When a Databricks feature might not work on Free Edition serverless, say so and give the fallback. Don't guess at platform behaviour as if it were fact.
- Ask before adding new dependencies.
- Writing style for the README and docstrings: plain British English, no em-dashes, no marketing language.

## Environment

- Local dev: VS Code on **WSL (Ubuntu)**, Python 3.11+, Java 21 for local PySpark (needs PySpark 4.x).
- Remote: Databricks **Free Edition**, serverless compute only, Unity Catalog `workspace` catalog, schema `workspace.fraud`.
- Code reaches Databricks through either the Databricks VS Code extension (Databricks Connect, serverless) or a Databricks **Git folder** linked to this repo. If extension auth fails, use the Git folder route: write and test locally, push, pull into the Git folder, run notebooks in the browser.
- Raw data lives in the UC volume `/Volumes/workspace/fraud/raw/` (`train_transaction.csv`, `train_identity.csv`). **Never commit data**: `data/` is gitignored, and the Kaggle rules forbid redistribution.

## Architecture

```
Kaggle CSVs -> UC volume -> bronze (raw Delta) -> silver (joined, clean, event_ts)
  -> gold (velocity + frequency features) -> train XGBoost on driver -> MLflow model (@champion)
  -> batch scoring (spark_udf) -> scores table -> weekly PSI drift table
```

Tables (all in `workspace.fraud`): `bronze_train_transaction`, `bronze_train_identity`, `silver_transactions`, `gold_features`, `scores`, `drift_psi`.
Registered model: `workspace.fraud.fraud_xgb`, alias `champion`.
The whole chain from bronze to drift runs as one scheduled Databricks Job.

## Repo structure

```
fraud/
  features.py      # add_velocity_features, add_frequency_features, CAT_COLS
  splits.py        # NON_FEATURES, feature_cols(), cutoffs(), with_split()
  drift.py         # bucket(), weekly_psi()
notebooks/
  01_bronze.py  02_silver.py  03_gold.py  04_train.py  05_score.py  06_drift.py
tests/
  conftest.py      # local SparkSession fixture (local[1])
  test_features.py
  test_splits.py
  test_drift.py
.github/workflows/tests.yml
requirements-dev.txt
README.md
CLAUDE.md
```

Logic lives in `fraud/` as pure functions (DataFrame in, DataFrame out) so it can be tested locally. Notebooks stay thin: read a table, call `fraud/` functions, write a table.

## Rules for the code

Data and leakage:
- Velocity windows use `rangeBetween(-secs, -1)` ordered by `TransactionDT`. They must only see strictly earlier transactions.
- Wrap any window keyed on a nullable column in a null guard (`F.when(col.isNotNull(), ...)`). Null keys must never share a history.
- The time split comes only from `fraud/splits.py`: train <70%, early_stop 70-80%, gap 80-85% (stands in for chargeback label delay), test >=85%. Quantiles are exact (`relativeError=0.0`).
- Early stopping and threshold choice use `early_stop`. The `test` split is touched only for final metrics.
- Frequency encodings currently use the full dataset (a small future-information leak, no labels). This is documented as a known limitation.

Spark:
- DataFrame API only. No RDDs, no `sparkContext` (not available on serverless).
- Use built-in functions over Python UDFs wherever possible.
- No `.cache()` / `.persist()`. Write intermediate results to Delta tables instead.
- `toPandas()` only on the finished, filtered feature table in `04_train`.
- Use `F.broadcast()` for small lookup tables (frequency maps).

Modelling:
- Cast features to `double` both in training and scoring, so they match the MLflow signature.
- Training and scoring get the feature list from `feature_cols()`. Never hand-maintain a second copy.
- Log params, `test_pr_auc`, `best_iteration` and the cost metric to MLflow. Register to Unity Catalog with `mlflow.set_registry_uri("databricks-uc")`.
- `xgboost.spark.SparkXGBClassifier` is optional and may not work on serverless. Driver training is the default.

Serverless:
- Install notebook libraries with `%pip install ...` then `dbutils.library.restartPython()`.
- If `mlflow.pyfunc.spark_udf` fails, fall back to `mapInPandas` with `mlflow.pyfunc.load_model`.

## Testing

```bash
pip install -r requirements-dev.txt   # pyspark, pytest
pytest -q
```

- Every function in `fraud/` gets a small hand-built DataFrame test with known expected values (for example, card_txn_1h for transactions at t=0, 1800, 5000 should be 0, 1, 1, and a null card1 gives null).
- Tests run on local PySpark only. They must never need a Databricks connection.
- CI runs `pytest` on every push via GitHub Actions.

## Build plan (tick off as we go)

- [x] Step 0: Download IEEE-CIS train files locally, gitignore `data/`
- [ ] Step 1: Free Edition workspace, schema + volume, upload CSVs, link Git folder
- [ ] Step 2: `01_bronze`: CSVs to Delta with `_ingested_at`, `_source_file`; check 590,540 / 144,233 rows
- [ ] Step 3: `02_silver`: column subset (no V columns yet), left join identity, `event_ts` (ref epoch 1512086400 = 2017-12-01 UTC, assumed), `event_day`, clean email domains
- [ ] Step 4: `fraud/features.py` + tests, `03_gold`; review `explain()` for Exchange operators
- [ ] Step 5: `fraud/splits.py` + tests, `04_train`: XGBoost, MLflow, register `@champion`  **(enough to apply)**
- [ ] Step 6: `fraud/drift.py` + tests, `05_score`, `06_drift`
- [ ] Step 7: Databricks Job chaining all six notebooks; screenshot DAG and run
- [ ] Step 8: README, CI workflow, cross-link with `fraud-detection-system`

Stretch (only after Step 8): customer ID key (card1 + addr1 + event_day - D1), frequency maps from the training window only, selected V columns, SparkXGBClassifier, Databricks Asset Bundle.

## README must cover

Purpose and link to the earlier repo; architecture diagram and Job screenshot; row counts and rough runtimes per stage; design decisions (time split and gap, past-only windows, null guard, broadcast joins, driver training); test-window PR-AUC and cost metric (not comparable with the old repo's 0.798, which used a different dataset and split); polars vs Spark trade-offs; known limitations (timestamp assumption, full-data frequency counts).

## Interview concepts I must be able to explain from this repo

Lazy evaluation; narrow vs wide transformations and shuffles; partitioning and AQE; broadcast joins; data skew and fixes; `rowsBetween` vs `rangeBetween`; risks of `toPandas`; Delta ACID, time travel, OPTIMIZE/clustering; MLflow aliases (champion/challenger); label delay and selective labels in fraud.
