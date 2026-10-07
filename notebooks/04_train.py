# Databricks notebook source
# MAGIC %md
# MAGIC # 04 Train: XGBoost on the driver, tracked in MLflow
# MAGIC
# MAGIC 1. Split `gold_features` by time (`fraud/splits.py`) and drop the `gap` rows.
# MAGIC 2. Bring the filtered feature table to the driver with `toPandas()`.
# MAGIC 3. Train XGBoost on `train`, with early stopping on `early_stop`.
# MAGIC 4. Choose the decision threshold that minimises cost on `early_stop`.
# MAGIC 5. Score `test` once for the final metrics.
# MAGIC 6. Log everything to MLflow, register the model in Unity Catalog, and give it the
# MAGIC    `champion` alias if it beats the current champion (otherwise `challenger`).

# COMMAND ----------

# MAGIC %pip install -q xgboost scikit-learn mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(".."))

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F
from sklearn.metrics import average_precision_score

from fraud.splits import cutoffs, feature_cols, with_split

CATALOG_SCHEMA = "workspace.fraud"
MODEL_NAME = f"{CATALOG_SCHEMA}.fraud_xgb"

# Illustrative unit costs, the same as the earlier fraud-detection-system repo: a missed
# fraud loses the transaction value on average, a false alarm costs review time and
# customer friction.
COST_FN = 100.0
COST_FP = 5.0

PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "early_stopping_rounds": 100,
    "random_state": 42,
    "n_jobs": -1,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split and bring to the driver
# MAGIC
# MAGIC `toPandas()` collects every selected row into the driver's memory. It is only safe on a
# MAGIC table that is already filtered and narrowed to what the model needs: here about 560k rows
# MAGIC by about 100 double columns, roughly 0.5 GB. On a much larger table the driver would run
# MAGIC out of memory, and the options would be sampling negatives, distributed training
# MAGIC (`SparkXGBClassifier`), or a bigger driver.
# MAGIC
# MAGIC Features are cast to `double` here and again in scoring, so the types always match the
# MAGIC model signature logged to MLflow. Spark nulls become `NaN`, which XGBoost treats as
# MAGIC missing and learns a default direction for at each split.

# COMMAND ----------

gold = spark.table(f"{CATALOG_SCHEMA}.gold_features")

cuts = cutoffs(gold)
features = feature_cols(gold)
print(cuts)
print(f"{len(features)} features")

model_df = (
    with_split(gold, cuts)
    .where(F.col("split") != "gap")
    .select(
        "split",
        F.col("isFraud").cast("int").alias("isFraud"),
        *[F.col(c).cast("double").alias(c) for c in features],
    )
)

pdf = model_df.toPandas()
print(pdf.groupby("split")["isFraud"].agg(["count", "mean"]))


def part(name):
    rows = pdf[pdf["split"] == name]
    return rows[features], rows["isFraud"].to_numpy()


X_train, y_train = part("train")
X_es, y_es = part("early_stop")
X_test, y_test = part("test")
del pdf

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train
# MAGIC
# MAGIC Trees are added until PR-AUC on `early_stop` has not improved for 100 rounds. Predictions
# MAGIC then use the best round, not the last.
# MAGIC
# MAGIC No class weighting (`scale_pos_weight`). Weighting shifts the predicted probabilities away
# MAGIC from the real fraud rate. Without it the scores stay roughly calibrated, and the
# MAGIC imbalance is handled where the business cost lives: the threshold.

# COMMAND ----------

model = xgb.XGBClassifier(**PARAMS)
model.fit(X_train, y_train, eval_set=[(X_es, y_es)], verbose=200)

best_iteration = model.best_iteration
print(f"best iteration: {best_iteration}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Threshold from `early_stop`
# MAGIC
# MAGIC For each candidate threshold, cost = missed frauds x 100 + false alarms x 5. The threshold
# MAGIC with the lowest cost on `early_stop` is fixed before the test set is looked at.
# MAGIC
# MAGIC For a perfectly calibrated model the best threshold would be
# MAGIC COST_FP / (COST_FP + COST_FN) = 5 / 105, about 0.048: flag whenever the expected loss from
# MAGIC letting it through (p x 100) exceeds the cost of a review ((1 - p) x 5). How far the chosen
# MAGIC threshold lands from 0.048 is a rough check of calibration.

# COMMAND ----------

def cost_at(y_true, scores, threshold):
    flagged = scores >= threshold
    fn = int(((y_true == 1) & ~flagged).sum())
    fp = int(((y_true == 0) & flagged).sum())
    return fn * COST_FN + fp * COST_FP, fn, fp


p_es = model.predict_proba(X_es)[:, 1]
thresholds = np.round(np.arange(0.005, 1.0, 0.005), 3)
es_costs = [cost_at(y_es, p_es, t)[0] for t in thresholds]
threshold = float(thresholds[int(np.argmin(es_costs))])

print(f"threshold: {threshold} (calibrated optimum would be {COST_FP / (COST_FP + COST_FN):.3f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test metrics
# MAGIC
# MAGIC The only time the test split is used. The naive baseline flags nothing, so it pays for
# MAGIC every fraud.

# COMMAND ----------

p_test = model.predict_proba(X_test)[:, 1]

test_cost, test_fn, test_fp = cost_at(y_test, p_test, threshold)
test_tp = int(y_test.sum()) - test_fn
baseline_cost = float(y_test.sum()) * COST_FN

metrics = {
    "early_stop_pr_auc": average_precision_score(y_es, p_es),
    "test_pr_auc": average_precision_score(y_test, p_test),
    "test_cost": test_cost,
    "test_baseline_cost": baseline_cost,
    "test_cost_reduction": 1 - test_cost / baseline_cost,
    "test_precision": test_tp / max(test_tp + test_fp, 1),
    "test_recall": test_tp / max(int(y_test.sum()), 1),
    "test_fraud_rate": float(y_test.mean()),
    "best_iteration": best_iteration,
}
for k, v in metrics.items():
    print(f"{k:<22}{v:,.4f}")

# COMMAND ----------

importance = (
    pd.Series(model.get_booster().get_score(importance_type="gain"))
    .sort_values(ascending=False)
    .head(20)
    .rename("gain")
    .reset_index()
    .rename(columns={"index": "feature"})
)
display(importance)

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow
# MAGIC
# MAGIC The model is wrapped in a small `pyfunc` class whose `predict` returns the fraud
# MAGIC probability. The plain XGBoost flavour's `predict` returns 0/1 labels at a fixed 0.5
# MAGIC cut, which is not what scoring needs.
# MAGIC
# MAGIC The experiment lives in the user's home folder: notebooks in a Git folder cannot hold their
# MAGIC own MLflow experiment.

# COMMAND ----------

class FraudProbability(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input, params=None):
        return pd.Series(self.model.predict_proba(model_input)[:, 1], name="fraud_probability")


user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/fraud-detection-databricks")
mlflow.set_registry_uri("databricks-uc")

signature = infer_signature(X_train.head(100), model.predict_proba(X_train.head(100))[:, 1])

with mlflow.start_run() as run:
    mlflow.log_params(PARAMS)
    mlflow.log_params({
        "n_features": len(features),
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
        "threshold": threshold,
        "cutoff_early_stop": cuts.early_stop,
        "cutoff_gap": cuts.gap,
        "cutoff_test": cuts.test,
    })
    mlflow.log_metrics(metrics)
    mlflow.log_dict({"features": features}, "features.json")
    mlflow.log_table(importance, "feature_importance.json")

    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=FraudProbability(model),
        signature=signature,
        input_example=X_train.head(5),
        extra_pip_requirements=[f"xgboost=={xgb.__version__}"],
        registered_model_name=MODEL_NAME,
    )

new_version = info.registered_model_version
print(f"run {run.info.run_id}, registered {MODEL_NAME} version {new_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Champion or challenger
# MAGIC
# MAGIC Aliases are movable pointers to a model version. Scoring always loads
# MAGIC `models:/workspace.fraud.fraud_xgb@champion`, so promoting a model is just moving the
# MAGIC alias, and rolling back is moving it back. No scoring code changes.
# MAGIC
# MAGIC The new version becomes `champion` only if its `early_stop` PR-AUC beats the current
# MAGIC champion's. Otherwise it gets the `challenger` alias. The comparison deliberately does not
# MAGIC use test PR-AUC: picking between models by their test score, run after run, would quietly
# MAGIC turn the test split into a validation set and make the reported test metric optimistic.

# COMMAND ----------

client = MlflowClient(registry_uri="databricks-uc")

try:
    current = client.get_model_version_by_alias(MODEL_NAME, "champion")
    current_pr_auc = client.get_run(current.run_id).data.metrics["early_stop_pr_auc"]
except MlflowException:  # no champion alias yet
    current, current_pr_auc = None, None

if current is None or metrics["early_stop_pr_auc"] > current_pr_auc:
    client.set_registered_model_alias(MODEL_NAME, "champion", new_version)
    outcome = "champion"
else:
    client.set_registered_model_alias(MODEL_NAME, "challenger", new_version)
    outcome = "challenger"

print(
    f"version {new_version} (early_stop PR-AUC {metrics['early_stop_pr_auc']:.4f}) is now {outcome}; "
    f"previous champion: {None if current is None else current.version} ({current_pr_auc})"
)
