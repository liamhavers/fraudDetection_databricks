from pyspark.sql import functions as F

from fraud.splits import Cutoffs, cutoffs, feature_cols, with_split


def split_counts(df):
    return {r.split: r["count"] for r in df.groupBy("split").count().collect()}


def test_cutoffs_are_exact_quantiles(spark):
    df = spark.range(1, 101).withColumnRenamed("id", "TransactionDT")

    assert cutoffs(df) == Cutoffs(early_stop=70, gap=80, test=85)


def test_split_sizes(spark):
    df = spark.range(1, 101).withColumnRenamed("id", "TransactionDT")
    out = with_split(df, cutoffs(df))

    assert split_counts(out) == {"train": 69, "early_stop": 10, "gap": 5, "test": 16}


def test_splits_are_in_time_order(spark):
    df = spark.range(1, 101).withColumnRenamed("id", "TransactionDT")
    out = with_split(df, cutoffs(df))
    bounds = {
        r.split: (r.lo, r.hi)
        for r in out.groupBy("split")
        .agg(F.min("TransactionDT").alias("lo"), F.max("TransactionDT").alias("hi"))
        .collect()
    }
    order = ["train", "early_stop", "gap", "test"]

    for earlier, later in zip(order, order[1:]):
        assert bounds[earlier][1] < bounds[later][0]


def test_same_timestamp_never_split(spark):
    df = spark.createDataFrame([(t,) for t in [1, 2, 3, 3, 3, 3, 4]], "TransactionDT int")
    out = with_split(df, Cutoffs(early_stop=3, gap=4, test=4))

    assert {r.split for r in out.where("TransactionDT = 3").collect()} == {"early_stop"}


def test_feature_cols(spark):
    df = spark.createDataFrame(
        [(1, 0, 100, 5, "a", 3, True, 1.5)],
        "TransactionID int, isFraud int, TransactionDT int, event_day int, "
        "card4 string, card4_freq long, has_identity boolean, TransactionAmt double",
    )

    assert feature_cols(df) == ["card4_freq", "has_identity", "TransactionAmt"]
