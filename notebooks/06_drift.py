# Databricks notebook source
# MAGIC %md
# MAGIC # 06 Drift: weekly PSI and performance
# MAGIC
# MAGIC Measures, for every week, how far the model's score and its 10 most important features
# MAGIC have moved from the training period, and writes the result to `drift_psi`.
# MAGIC
# MAGIC - **Score PSI** answers "has the model's output changed?"
# MAGIC - **Feature PSI** answers "which inputs moved?"
# MAGIC - **Weekly precision and recall** answer "did it matter?". In production these lag by the
# MAGIC   chargeback delay, which is why PSI, which needs no labels, is the early warning.
# MAGIC
# MAGIC Reference period: the `train` split. PSI rule of thumb: below 0.1 stable, 0.1 to 0.25
# MAGIC moderate, above 0.25 major.

# COMMAND ----------

# MAGIC %pip install -q mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(".."))

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

from fraud.drift import quantile_edges, weekly_psi, with_week

CATALOG_SCHEMA = "workspace.fraud"
MODEL_NAME = f"{CATALOG_SCHEMA}.fraud_xgb"
TOP_N_FEATURES = 10

# Week 0 starts on the assumed reference date (see 02_silver). Only used for readable labels.
REF_DATE = "2017-12-01"

# COMMAND ----------

client = MlflowClient(registry_uri="databricks-uc")
champion = client.get_model_version_by_alias(MODEL_NAME, "champion")

# Read the importance file logged by 04_train directly. mlflow.load_table finds tables through
# a run tag that is not always set on serverless, so it can miss an artifact that exists.
logged = mlflow.artifacts.load_dict(f"runs:/{champion.run_id}/feature_importance.json")
importance = pd.DataFrame(logged["data"], columns=logged["columns"])
top_features = importance.sort_values("gain", ascending=False)["feature"].head(TOP_N_FEATURES).tolist()
print(f"champion version {champion.version}; monitoring score and {top_features}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## PSI
# MAGIC
# MAGIC Score PSI comes from `scores` and feature PSI from `gold_features`. Both use the same
# MAGIC `week` and the same reference rows (`split = 'train'`), so no join between the two tables
# MAGIC is needed. Joining two 590k-row tables would mean shuffling both sides (a sort-merge join).

# COMMAND ----------

scores = spark.table(f"{CATALOG_SCHEMA}.scores")

train_ids = scores.where(F.col("split") == "train").select("TransactionID")
gold = with_week(spark.table(f"{CATALOG_SCHEMA}.gold_features")).select(
    "TransactionID", "week", *[F.col(c).cast("double").alias(c) for c in top_features]
)
# Semi join: keeps gold rows whose TransactionID is in the train split, adds no columns.
gold_ref = gold.join(train_ids, "TransactionID", "left_semi")
score_ref = scores.where(F.col("split") == "train")

score_edges = {"fraud_probability": quantile_edges(score_ref, "fraud_probability")}
feature_edges = {c: quantile_edges(gold_ref, c) for c in top_features}

psi = weekly_psi(scores, score_ref, score_edges).unionByName(
    weekly_psi(gold, gold_ref, feature_edges)
)

drift = psi.select(
    "week",
    F.date_add(F.lit(REF_DATE).cast("date"), F.col("week") * 7).alias("week_start"),
    "feature",
    "rows",
    "psi",
    F.when(F.col("psi") < 0.1, "stable")
    .when(F.col("psi") < 0.25, "moderate")
    .otherwise("major")
    .alias("status"),
    F.lit(int(champion.version)).alias("model_version"),
    F.current_timestamp().alias("computed_at"),
)

(
    drift.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable(f"{CATALOG_SCHEMA}.drift_psi")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results
# MAGIC
# MAGIC Weeks 0 and 26 are partial (days 1 to 6 and 182 to 183), so their PSI is noisier. Check
# MAGIC `rows` before reading much into them.

# COMMAND ----------

d = spark.table(f"{CATALOG_SCHEMA}.drift_psi")

display(
    d.groupBy("week", "week_start", "rows")
    .pivot("feature", ["fraud_probability", *top_features])
    .agg(F.round(F.first("psi"), 3))
    .orderBy("week")
)

# COMMAND ----------

display(
    d.where(F.col("status") != "stable")
    .orderBy(F.desc("psi"))
    .select("week", "week_start", "feature", "rows", "psi", "status")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Weekly performance
# MAGIC
# MAGIC Precision and recall at the champion's threshold, by week. Compare the weeks with high score
# MAGIC PSI above against any drop here.

# COMMAND ----------

# try_divide returns null instead of failing when a week has no flags or no fraud
# (serverless runs with ANSI SQL mode, where a plain division by zero is an error).
caught = F.sum((F.col("flagged") & (F.col("isFraud") == 1)).cast("int"))

display(
    scores.groupBy("week", "split")
    .agg(
        F.count("*").alias("rows"),
        F.avg("isFraud").alias("fraud_rate"),
        F.avg(F.col("flagged").cast("int")).alias("flag_rate"),
        F.try_divide(caught, F.sum(F.col("flagged").cast("int"))).alias("precision"),
        F.try_divide(caught, F.sum("isFraud")).alias("recall"),
    )
    .orderBy("week")
)
