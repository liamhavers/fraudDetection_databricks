# Databricks notebook source
# MAGIC %md
# MAGIC # 05 Score: batch scoring with the champion model
# MAGIC
# MAGIC 1. Look up the version behind `workspace.fraud.fraud_xgb@champion` and read its threshold
# MAGIC    and split cutoffs from its MLflow run.
# MAGIC 2. Check the feature list matches the model signature.
# MAGIC 3. Score every row of `gold_features` in Spark and write `scores`.
# MAGIC 4. Recompute test PR-AUC from `scores` and check it matches the training run.
# MAGIC
# MAGIC Every row is scored, including the training period, because `06_drift` uses the training
# MAGIC period's scores as its reference. A production job would score only new transactions.

# COMMAND ----------

# MAGIC %pip install -q xgboost scikit-learn mlflow

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
from sklearn.metrics import average_precision_score

from fraud.drift import with_week
from fraud.splits import Cutoffs, feature_cols, with_split

CATALOG_SCHEMA = "workspace.fraud"
MODEL_NAME = f"{CATALOG_SCHEMA}.fraud_xgb"
MODEL_URI = f"models:/{MODEL_NAME}@champion"

# mlflow.pyfunc.spark_udf is the default. If it fails on serverless, set this to False to use
# the mapInPandas fallback below.
USE_SPARK_UDF = True

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient(registry_uri="databricks-uc")

champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
run = client.get_run(champion.run_id)
params = run.data.params

threshold = float(params["threshold"])
cuts = Cutoffs(
    early_stop=float(params["cutoff_early_stop"]),
    gap=float(params["cutoff_gap"]),
    test=float(params["cutoff_test"]),
)
print(f"champion: version {champion.version}, threshold {threshold}, {cuts}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature check
# MAGIC
# MAGIC Training and scoring both take the feature list from `feature_cols()`. This check fails
# MAGIC loudly if `gold_features` has changed shape since the champion was trained (a column added,
# MAGIC removed or reordered), instead of scoring with misaligned inputs. Mismatched features
# MAGIC between training and serving (training-serving skew) are a common silent failure.

# COMMAND ----------

gold = spark.table(f"{CATALOG_SCHEMA}.gold_features")
features = feature_cols(gold)

model_inputs = mlflow.models.get_model_info(MODEL_URI).signature.inputs.input_names()
assert features == model_inputs, (
    f"feature mismatch: only in gold {sorted(set(features) - set(model_inputs))}, "
    f"only in model {sorted(set(model_inputs) - set(features))}"
)

to_score = with_week(with_split(gold, cuts)).select(
    "TransactionID",
    "event_ts",
    "week",
    "split",
    F.col("isFraud").cast("int").alias("isFraud"),
    *[F.col(c).cast("double").alias(c) for c in features],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score
# MAGIC
# MAGIC **`spark_udf`** wraps the model in a pandas UDF. Spark sends each partition to Python in
# MAGIC Arrow batches, the model scores a batch at a time, and the work is spread across executors.
# MAGIC
# MAGIC **`mapInPandas` fallback**: the model is loaded once on the driver and shipped to the
# MAGIC executors inside the function's closure. Each executor gets an iterator of pandas batches
# MAGIC for its partition and yields scored batches back.
# MAGIC
# MAGIC Both are Python UDFs, so they are slower than built-in functions: rows have to cross from the
# MAGIC JVM to Python and back. There is no built-in way to run an XGBoost model, so a UDF is the
# MAGIC right tool here.

# COMMAND ----------

def score_with_spark_udf(df):
    predict = mlflow.pyfunc.spark_udf(spark, MODEL_URI, result_type="double", env_manager="local")
    return df.withColumn("fraud_probability", predict(F.struct(*features)))


def score_with_map_in_pandas(df):
    model = mlflow.pyfunc.load_model(MODEL_URI)
    out_schema = "TransactionID int, fraud_probability double"

    def predict_batches(batches):
        for batch in batches:
            yield pd.DataFrame({
                "TransactionID": batch["TransactionID"],
                "fraud_probability": model.predict(batch[features]).to_numpy(),
            })

    scored = df.select("TransactionID", *features).mapInPandas(predict_batches, out_schema)
    return df.join(scored, "TransactionID")


score = score_with_spark_udf if USE_SPARK_UDF else score_with_map_in_pandas

scores = score(to_score).select(
    "TransactionID",
    "event_ts",
    "week",
    "split",
    "isFraud",
    "fraud_probability",
    (F.col("fraud_probability") >= F.lit(threshold)).alias("flagged"),
    F.lit(threshold).alias("threshold"),
    F.lit(int(champion.version)).alias("model_version"),
    F.current_timestamp().alias("scored_at"),
)

(
    scores.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable(f"{CATALOG_SCHEMA}.scores")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checks
# MAGIC
# MAGIC - One score per transaction, none null.
# MAGIC - Test PR-AUC recomputed from the `scores` table matches the training run. If scoring
# MAGIC   prepared features differently from training, this is where it would show.

# COMMAND ----------

s = spark.table(f"{CATALOG_SCHEMA}.scores")

stats = s.agg(
    F.count("*").alias("rows"),
    F.countDistinct("TransactionID").alias("ids"),
    F.sum(F.col("fraud_probability").isNull().cast("int")).alias("null_scores"),
).first()
print(stats.asDict())
assert stats.rows == stats.ids == 590_540, "expected one score per transaction"
assert stats.null_scores == 0, "null scores"

test = s.where(F.col("split") == "test").select("isFraud", "fraud_probability").toPandas()
scored_pr_auc = average_precision_score(test["isFraud"], test["fraud_probability"])
trained_pr_auc = run.data.metrics["test_pr_auc"]
print(f"test PR-AUC from scores: {scored_pr_auc:.6f}, from training run: {trained_pr_auc:.6f}")
assert abs(scored_pr_auc - trained_pr_auc) < 1e-4, "scoring does not reproduce training"

# COMMAND ----------

display(
    s.groupBy("split")
    .agg(
        F.count("*").alias("rows"),
        F.avg(F.col("flagged").cast("int")).alias("flag_rate"),
        F.avg("isFraud").alias("fraud_rate"),
        F.avg("fraud_probability").alias("mean_score"),
    )
    .orderBy("split")
)
