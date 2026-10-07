# Databricks notebook source
# MAGIC %md
# MAGIC # 03 Gold: model features
# MAGIC
# MAGIC Adds two kinds of feature to silver and writes `gold_features`:
# MAGIC
# MAGIC 1. **Frequency encodings** (`{col}_freq`) for the categorical columns in `CAT_COLS`.
# MAGIC 2. **Velocity features** per `card1`: counts and amounts of earlier transactions in the last
# MAGIC    1 hour, 24 hours and 7 days, and seconds since the previous transaction.
# MAGIC
# MAGIC The logic lives in `fraud/features.py` and is unit tested locally. This notebook only reads,
# MAGIC calls those functions, checks and writes.
# MAGIC
# MAGIC Known limitation: frequency counts use the full dataset, including the test period. No labels
# MAGIC are involved, but it is a small leak of future information.

# COMMAND ----------

import os
import sys

# The repo root is the parent of this notebook's folder. Adding it to the path lets the
# notebook import the `fraud` package from the Git folder.
sys.path.append(os.path.abspath(".."))

from pyspark.sql import functions as F

from fraud.features import CAT_COLS, add_frequency_features, add_velocity_features

CATALOG_SCHEMA = "workspace.fraud"
EXPECTED_ROWS = 590_540

# COMMAND ----------

silver = spark.table(f"{CATALOG_SCHEMA}.silver_transactions")

gold = add_velocity_features(add_frequency_features(silver), key="card1", prefix="card")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Skew check
# MAGIC
# MAGIC The velocity windows partition by `card1`, so every row for one `card1` value is processed
# MAGIC by a single task. If one value held most of the data, that task would run long after the
# MAGIC others finished (skew). AQE can split skewed partitions in joins, but not in window
# MAGIC functions, so this is worth checking by hand.

# COMMAND ----------

display(
    silver.groupBy("card1")
    .count()
    .withColumn("share", F.col("count") / F.lit(EXPECTED_ROWS))
    .orderBy(F.desc("count"))
    .limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query plan
# MAGIC
# MAGIC What to look for:
# MAGIC
# MAGIC - **One** `Exchange hashpartitioning(card1, ...)` feeding a sort and the `Window` operators.
# MAGIC   All seven velocity columns share the same `partitionBy` and `orderBy`, so they should share
# MAGIC   one shuffle and one sort.
# MAGIC - For each frequency column: a small `Exchange` for the `groupBy` (after a partial aggregate,
# MAGIC   so only one row per distinct value per partition is shuffled), then a broadcast exchange
# MAGIC   and a `BroadcastHashJoin`. The large side is not shuffled for these joins.

# COMMAND ----------

gold.explain(mode="formatted")

# COMMAND ----------

(
    gold.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable(f"{CATALOG_SCHEMA}.gold_features")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checks
# MAGIC
# MAGIC - Row count unchanged: the frequency joins are left joins on a grouped table, so each key
# MAGIC   matches at most one row.
# MAGIC - Every frequency column is null only where its source column is null.
# MAGIC - Each card's first transaction has `card_txn_7d = 0` and a null `card_secs_since_last`.

# COMMAND ----------

g = spark.table(f"{CATALOG_SCHEMA}.gold_features")

rows = g.count()
assert rows == EXPECTED_ROWS, f"expected {EXPECTED_ROWS:,} rows, got {rows:,}"

mismatch = g.agg(
    *[
        F.sum((F.col(c).isNull() != F.col(f"{c}_freq").isNull()).cast("int")).alias(c)
        for c in CAT_COLS
    ]
).first().asDict()
bad = {c: n for c, n in mismatch.items() if n}
assert not bad, f"frequency nulls do not match source nulls: {bad}"

first_txn = g.where(F.col("card_secs_since_last").isNull()).agg(
    F.count("*").alias("cards"),
    F.max("card_txn_7d").alias("max_txn_7d"),
).first()
print(f"rows: {rows:,}; first transactions per card: {first_txn.cards:,}")
assert first_txn.max_txn_7d == 0, "a first transaction has earlier transactions in its window"

# COMMAND ----------

display(
    g.groupBy("isFraud").agg(
        *[
            F.expr(f"percentile_approx({c}, 0.5)").alias(f"median_{c}")
            for c in ["card_txn_1h", "card_txn_24h", "card_txn_7d", "card_secs_since_last", "card1_freq"]
        ]
    )
)
