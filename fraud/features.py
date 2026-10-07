"""Feature engineering for the gold table.

Every function takes a DataFrame and returns a DataFrame with extra columns, so it can be
tested locally on a small hand-built DataFrame.
"""

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

VELOCITY_WINDOWS = {"1h": 3_600, "24h": 86_400, "7d": 604_800}

# Columns to frequency-encode: every string column in silver, plus integer columns that are
# category codes rather than quantities.
CAT_COLS = [
    "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2",
    "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29", "id_30", "id_31",
    "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
    "DeviceType", "DeviceInfo",
]


def _null_guard(key: str, value: Column) -> Column:
    """Return value only where the key is present.

    Spark puts every null key into one window partition, so without this guard all
    transactions with a missing key would share one made-up history.
    """
    return F.when(F.col(key).isNotNull(), value)


def add_velocity_features(
    df: DataFrame,
    key: str,
    prefix: str,
    windows: dict[str, int] = VELOCITY_WINDOWS,
    ts_col: str = "TransactionDT",
    amt_col: str = "TransactionAmt",
) -> DataFrame:
    """Add past-only transaction counts and amounts per key.

    For each window name and length in seconds, adds:
      {prefix}_txn_{name}: number of earlier transactions for the key in [t - secs, t - 1]
      {prefix}_amt_{name}: total amount of those transactions (0.0 if none)
    and also:
      {prefix}_secs_since_last: seconds since the key's previous transaction (null if none)

    The frame ends at -1, so a transaction never sees itself or anything in the same second.
    Rows with a null key get null for every feature.
    """
    by_key = Window.partitionBy(key).orderBy(ts_col)

    for name, secs in windows.items():
        w = by_key.rangeBetween(-secs, -1)
        df = df.withColumn(
            f"{prefix}_txn_{name}", _null_guard(key, F.count(F.lit(1)).over(w))
        ).withColumn(
            f"{prefix}_amt_{name}",
            _null_guard(key, F.coalesce(F.sum(amt_col).over(w), F.lit(0.0))),
        )

    all_earlier = by_key.rangeBetween(Window.unboundedPreceding, -1)
    return df.withColumn(
        f"{prefix}_secs_since_last",
        _null_guard(key, F.col(ts_col) - F.max(ts_col).over(all_earlier)),
    )


def add_frequency_features(df: DataFrame, cols: list[str] = CAT_COLS) -> DataFrame:
    """Add {col}_freq: how many rows in df share this row's value of col.

    Counts come from the whole of df, so if df spans the test period they use future rows
    (no labels). Rows where col is null get a null count.

    Each count table has one row per distinct value, so it is broadcast to the join rather
    than shuffling df.
    """
    out = df
    for c in cols:
        counts = (
            df.where(F.col(c).isNotNull())
            .groupBy(c)
            .agg(F.count(F.lit(1)).alias(f"{c}_freq"))
        )
        out = out.join(F.broadcast(counts), on=c, how="left")

    return out.select(*df.columns, *[f"{c}_freq" for c in cols])
