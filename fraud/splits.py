"""Time-based split and the feature list shared by training and scoring.

Splits by TransactionDT quantile, never at random:

  train       before the 70% quantile
  early_stop  70% to 80%   early stopping and threshold choice
  gap         80% to 85%   unused, stands in for chargeback label delay
  test        from 85%     final metrics only
"""

from typing import NamedTuple

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, NumericType

TS_COL = "TransactionDT"

# Never used as model inputs. TransactionDT and event_day are absolute time: the test period
# has values the model never saw in training, so a tree could only learn "when", not "what".
NON_FEATURES = {
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "event_ts",
    "event_day",
    "split",
}


class Cutoffs(NamedTuple):
    """First TransactionDT of each split after train."""

    early_stop: float
    gap: float
    test: float


def feature_cols(df: DataFrame) -> list[str]:
    """Numeric and boolean columns of df that are not in NON_FEATURES, in column order.

    String columns are left out: the model sees them through their {col}_freq encodings.
    """
    return [
        f.name
        for f in df.schema.fields
        if f.name not in NON_FEATURES
        and isinstance(f.dataType, (NumericType, BooleanType))
    ]


def cutoffs(df: DataFrame, ts_col: str = TS_COL) -> Cutoffs:
    """Exact 70%, 80% and 85% quantiles of ts_col.

    relativeError=0.0 makes approxQuantile exact. It costs more than an approximate
    quantile, but a split that moved between runs would make results irreproducible.
    """
    q70, q80, q85 = df.approxQuantile(ts_col, [0.70, 0.80, 0.85], 0.0)
    return Cutoffs(early_stop=q70, gap=q80, test=q85)


def split_col(cuts: Cutoffs, ts_col: str = TS_COL) -> Column:
    ts = F.col(ts_col)
    return (
        F.when(ts < cuts.early_stop, "train")
        .when(ts < cuts.gap, "early_stop")
        .when(ts < cuts.test, "gap")
        .otherwise("test")
    )


def with_split(df: DataFrame, cuts: Cutoffs, ts_col: str = TS_COL) -> DataFrame:
    """Add a `split` column. Rows with the same timestamp always land in the same split."""
    return df.withColumn("split", split_col(cuts, ts_col))
