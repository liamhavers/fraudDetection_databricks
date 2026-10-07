# Databricks notebook source
# MAGIC %md
# MAGIC # 02 Silver: joined, cleaned transactions
# MAGIC
# MAGIC One row per transaction, built from the two bronze tables:
# MAGIC
# MAGIC 1. Drop the V1 to V339 columns (out of scope for now) and the bronze lineage columns.
# MAGIC 2. Cast integer-coded columns that the CSV stored as `315.0` back to `int`.
# MAGIC 3. Left join identity on `TransactionID`, with a `has_identity` flag.
# MAGIC 4. Add `event_ts` and `event_day` from `TransactionDT`.
# MAGIC 5. Clean the purchaser and recipient email domains.

# COMMAND ----------

import re

from pyspark.sql import functions as F

CATALOG_SCHEMA = "workspace.fraud"
EXPECTED_ROWS = 590_540
EXPECTED_IDENTITY_ROWS = 144_233

# TransactionDT is seconds from an unknown reference point. 2017-12-01 00:00 UTC is the
# community's assumed start date, based on holiday spikes in the data. Not confirmed by Vesta.
REF_EPOCH = 1_512_086_400

# Category codes the CSV wrote as floats (e.g. 315.0) because they contain nulls.
INT_CODED_COLS = ["card2", "card3", "card5", "addr1", "addr2"]

# COMMAND ----------

txn = spark.table(f"{CATALOG_SCHEMA}.bronze_train_transaction")
idn = spark.table(f"{CATALOG_SCHEMA}.bronze_train_identity")

txn_cols = [
    c for c in txn.columns
    if not re.fullmatch(r"V\d+", c) and not c.startswith("_")
]
idn_cols = [c for c in idn.columns if not c.startswith("_")]

print(f"transaction: {len(txn.columns)} bronze columns -> {len(txn_cols)} kept")
print(f"identity: {len(idn.columns)} bronze columns -> {len(idn_cols)} kept")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join
# MAGIC
# MAGIC A left join keeps every transaction. Only about a quarter have an identity record, and in the
# MAGIC earlier pandas project the fraud rate differed sharply between the two groups, so whether a
# MAGIC match exists is recorded as `has_identity` rather than lost.
# MAGIC
# MAGIC A join is a wide transformation: rows with the same key must end up on the same executor.
# MAGIC Spark can do that in two main ways:
# MAGIC
# MAGIC - **Sort-merge join**: shuffle both sides by `TransactionID` across the network, sort, merge.
# MAGIC - **Broadcast hash join**: send a full copy of the small side to every executor, so the large
# MAGIC   side is never shuffled.
# MAGIC
# MAGIC No `F.broadcast()` hint is needed. Identity is 27 MB as CSV, but as a Delta table it is
# MAGIC compressed Parquet with full table statistics, so the optimizer knows its real size and picks
# MAGIC a broadcast hash join in the initial plan. If the statistics were missing, Adaptive Query
# MAGIC Execution (AQE) could still switch to a broadcast join at runtime once the scan had measured
# MAGIC the real size. The `explain()` cell below shows the plan Spark chose.

# COMMAND ----------

identity = idn.select(*idn_cols).withColumn("has_identity", F.lit(True))

joined = (
    txn.select(*txn_cols)
    .join(identity, on="TransactionID", how="left")
    .withColumn("has_identity", F.coalesce(F.col("has_identity"), F.lit(False)))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean and derive columns
# MAGIC
# MAGIC - `event_ts`: reference epoch plus `TransactionDT` seconds, as a real timestamp.
# MAGIC - `event_day`: whole days since the reference point, `floor(TransactionDT / 86400)`, as an
# MAGIC   integer. Computed from the raw seconds rather than `to_date(event_ts)`, so it does not depend
# MAGIC   on the session time zone. An integer day also lines up with the `D` columns (days since an
# MAGIC   earlier event), which the stretch customer key (`event_day - D1`) needs.
# MAGIC - Email domains: lower case, trimmed, empty strings to null, and `gmail` (no suffix) merged
# MAGIC   into `gmail.com`. Other variants such as `yahoo.co.jp` are kept as they are, since the
# MAGIC   country suffix may carry signal.
# MAGIC
# MAGIC All of these are narrow transformations: each output row depends on one input row, so no
# MAGIC shuffle is needed.

# COMMAND ----------

def clean_email_domain(col):
    cleaned = F.lower(F.trim(col))
    return (
        F.when(cleaned.isNull() | (cleaned == ""), None)
        .when(cleaned == "gmail", "gmail.com")
        .otherwise(cleaned)
    )


silver = (
    joined
    .select(
        *[F.col(c).cast("int").alias(c) if c in INT_CODED_COLS else F.col(c) for c in joined.columns]
    )
    .withColumn("event_ts", F.timestamp_seconds(F.lit(REF_EPOCH) + F.col("TransactionDT")))
    .withColumn("event_day", F.floor(F.col("TransactionDT") / 86_400).cast("int"))
    .withColumn("P_emaildomain", clean_email_domain(F.col("P_emaildomain")))
    .withColumn("R_emaildomain", clean_email_domain(F.col("R_emaildomain")))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query plan
# MAGIC
# MAGIC What the plan shows (operators are prefixed `Photon`, Databricks' vectorised engine):
# MAGIC
# MAGIC - `PhotonBroadcastHashJoin LeftOuter`: identity is broadcast, transactions are joined in place.
# MAGIC - The only exchange is on the identity side, with `SinglePartition` and `EXECUTOR_BROADCAST`.
# MAGIC   That is how Photon builds the broadcast copy, not a hash-partitioned shuffle. The
# MAGIC   transaction side has no exchange at all, which is the point of a broadcast join.
# MAGIC - The transaction scan reads 55 columns, not 394. Parquet is columnar, so the dropped V
# MAGIC   columns are never read from storage (column pruning).
# MAGIC - `RequiredDataFilters: [isnotnull(TransactionID)]` on identity: a null key can never match
# MAGIC   in an equi-join, so the optimizer filters it out before the join.
# MAGIC
# MAGIC With AQE on, this is the initial plan (`isFinalPlan=false`). The final plan is in the query
# MAGIC profile for the write in the next cell.

# COMMAND ----------

silver.explain(mode="formatted")

# COMMAND ----------

(
    silver.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable(f"{CATALOG_SCHEMA}.silver_transactions")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checks
# MAGIC
# MAGIC - Row count unchanged by the left join, which also proves `TransactionID` is unique in identity
# MAGIC   (a duplicate key would add rows).
# MAGIC - Every identity row matched a transaction.
# MAGIC - No null `event_ts` or `event_day`.

# COMMAND ----------

s = spark.table(f"{CATALOG_SCHEMA}.silver_transactions")

stats = s.agg(
    F.count("*").alias("rows"),
    F.countDistinct("TransactionID").alias("distinct_ids"),
    F.sum(F.col("has_identity").cast("int")).alias("with_identity"),
    F.sum(F.col("event_ts").isNull().cast("int")).alias("null_event_ts"),
    F.sum(F.col("event_day").isNull().cast("int")).alias("null_event_day"),
    F.min("event_ts").alias("first_ts"),
    F.max("event_ts").alias("last_ts"),
    F.avg("isFraud").alias("fraud_rate"),
).first()

print(stats.asDict())

assert stats.rows == EXPECTED_ROWS, f"expected {EXPECTED_ROWS:,} rows, got {stats.rows:,}"
assert stats.distinct_ids == EXPECTED_ROWS, "TransactionID is not unique"
assert stats.with_identity == EXPECTED_IDENTITY_ROWS, (
    f"expected {EXPECTED_IDENTITY_ROWS:,} identity matches, got {stats.with_identity:,}"
)
assert stats.null_event_ts == 0 and stats.null_event_day == 0, "null event_ts or event_day"

# COMMAND ----------

display(
    s.groupBy("has_identity")
    .agg(F.count("*").alias("rows"), F.avg("isFraud").alias("fraud_rate"))
)

# COMMAND ----------

display(
    s.groupBy("P_emaildomain").count().orderBy(F.desc("count")).limit(20)
)
