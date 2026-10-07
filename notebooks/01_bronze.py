# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Bronze: raw CSVs to Delta
# MAGIC
# MAGIC Reads the two IEEE-CIS training CSVs from the Unity Catalog volume and writes each one,
# MAGIC unchanged apart from two lineage columns, to a Delta table:
# MAGIC
# MAGIC - `_ingested_at`: when this load ran
# MAGIC - `_source_file`: the file each row came from
# MAGIC
# MAGIC Bronze is a faithful copy of the source. No filtering, joining or renaming happens here,
# MAGIC so any later stage can be rebuilt from these tables without going back to the CSVs.

# COMMAND ----------

from pyspark.sql import functions as F

CATALOG_SCHEMA = "workspace.fraud"
RAW_DIR = "/Volumes/workspace/fraud/raw"

SOURCES = {
    "bronze_train_transaction": ("train_transaction.csv", 590_540),
    "bronze_train_identity": ("train_identity.csv", 144_233),
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read and write
# MAGIC
# MAGIC `inferSchema=True` makes Spark scan each file once to work out column types, then read it
# MAGIC again to load it. That costs an extra pass, but this is a one-off load of under 1 GB and
# MAGIC it saves hand-writing a schema for 394 transaction columns. In a production feed with a
# MAGIC fixed contract you would pass an explicit schema instead, so a bad file fails loudly rather
# MAGIC than silently changing a column's type.
# MAGIC
# MAGIC `_metadata.file_path` is the Unity Catalog way to get the source file. The older
# MAGIC `input_file_name()` is not supported on serverless / shared compute.
# MAGIC
# MAGIC Nothing actually runs until `saveAsTable`: the read, `withColumn` calls and write form one
# MAGIC plan (lazy evaluation). The schema inference scan is the exception, as it runs when the
# MAGIC reader is created.

# COMMAND ----------

for table, (filename, _) in SOURCES.items():
    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(f"{RAW_DIR}/{filename}")
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_file", F.col("_metadata.file_path"))
    )
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", True)
        .saveAsTable(f"{CATALOG_SCHEMA}.{table}")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checks
# MAGIC
# MAGIC Row counts must match the known Kaggle totals. The job should fail here rather than pass
# MAGIC a partial load on to silver.

# COMMAND ----------

for table, (_, expected) in SOURCES.items():
    actual = spark.table(f"{CATALOG_SCHEMA}.{table}").count()
    print(f"{table}: {actual:,} rows (expected {expected:,})")
    assert actual == expected, f"{table}: expected {expected:,} rows, got {actual:,}"

# COMMAND ----------

display(spark.table(f"{CATALOG_SCHEMA}.bronze_train_transaction").select(
    "TransactionID", "TransactionDT", "TransactionAmt", "isFraud", "_ingested_at", "_source_file"
).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delta history
# MAGIC
# MAGIC Each overwrite creates a new table version. Older versions stay readable with
# MAGIC `VERSION AS OF` (time travel) until `VACUUM` removes their files.

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY {CATALOG_SCHEMA}.bronze_train_transaction"))
