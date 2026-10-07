"""Population Stability Index (PSI) per week, against a reference period.

PSI compares how values are spread over fixed buckets in two samples:

  PSI = sum over buckets of (actual% - reference%) * ln(actual% / reference%)

Common rule of thumb: below 0.1 stable, 0.1 to 0.25 moderate shift, above 0.25 major shift.

Bucket edges come from the reference period only (its quantiles), so the reference is spread
roughly evenly over the buckets and every later week is measured against the same edges.
Nulls get their own bucket, because a change in how often a value is missing is drift too.
"""

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

NULL_BUCKET = -1

# Floor on bucket proportions, so an empty bucket gives a large but finite PSI term
# instead of ln(0).
EPSILON = 1e-4


def quantile_edges(df: DataFrame, col: str, n_buckets: int = 10) -> list[float]:
    """Inner bucket edges at the exact 1/n, 2/n, ... quantiles of col, ignoring nulls.

    Repeated edges are dropped, so a column with heavy ties (a count that is mostly 0, say)
    gets fewer buckets rather than empty ones.
    """
    probs = [i / n_buckets for i in range(1, n_buckets)]
    edges = df.approxQuantile(col, probs, 0.0)
    return sorted(set(edges))


def bucket(col: str, edges: list[float]) -> Column:
    """Bucket index of col: the number of edges at or below the value, so 0 to len(edges).

    Null values go to NULL_BUCKET.
    """
    index = sum((F.col(col) >= F.lit(e)).cast("int") for e in edges) if edges else F.lit(0)
    return F.when(F.col(col).isNull(), F.lit(NULL_BUCKET)).otherwise(index)


def _long_buckets(df: DataFrame, edges: dict[str, list[float]], keep: list[str]) -> DataFrame:
    """One row per input row and column: keep columns, feature name, bucket index."""
    pairs = F.array(*[
        F.struct(F.lit(name).alias("feature"), bucket(name, e).alias("bucket"))
        for name, e in edges.items()
    ])
    return df.select(*keep, F.explode(pairs).alias("fb")).select(*keep, "fb.feature", "fb.bucket")


def weekly_psi(
    df: DataFrame,
    reference: DataFrame,
    edges: dict[str, list[float]],
    week_col: str = "week",
) -> DataFrame:
    """PSI of each column in edges, for each week of df, against reference.

    Returns one row per (week, feature) with columns week, feature, rows, psi.
    """
    ref = (
        _long_buckets(reference, edges, keep=[])
        .groupBy("feature", "bucket")
        .count()
        .withColumn("ref_p", F.col("count") / F.sum("count").over(Window.partitionBy("feature")))
        .drop("count")
    )

    actual = (
        _long_buckets(df, edges, keep=[week_col])
        .groupBy(week_col, "feature", "bucket")
        .count()
    )

    # Every (week, feature, bucket) seen in either sample, so a bucket that is empty on one
    # side still contributes to PSI.
    weeks = df.select(week_col).distinct()
    grid = (
        weeks.crossJoin(ref.select("feature", "bucket"))
        .unionByName(actual.select(week_col, "feature", "bucket"))
        .distinct()
    )

    per_week = Window.partitionBy(week_col, "feature")
    return (
        grid.join(actual, [week_col, "feature", "bucket"], "left")
        .join(F.broadcast(ref), ["feature", "bucket"], "left")
        .fillna(0, subset=["count", "ref_p"])
        .withColumn("rows", F.sum("count").over(per_week))
        .withColumn("p", F.greatest(F.col("count") / F.col("rows"), F.lit(EPSILON)))
        .withColumn("q", F.greatest(F.col("ref_p"), F.lit(EPSILON)))
        .groupBy(week_col, "feature")
        .agg(
            F.first("rows").alias("rows"),
            F.sum((F.col("p") - F.col("q")) * F.log(F.col("p") / F.col("q"))).alias("psi"),
        )
    )
