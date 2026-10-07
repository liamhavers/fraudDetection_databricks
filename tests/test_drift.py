import math

import pytest

from fraud.drift import NULL_BUCKET, bucket, quantile_edges, weekly_psi, with_week


def psi_by_week(spark, ref_values, weeks):
    """weeks: {week: [values]}. Uses a single edge at 10, so buckets are <10 and >=10."""
    ref = spark.createDataFrame([(v,) for v in ref_values], "x double")
    df = spark.createDataFrame(
        [(w, v) for w, values in weeks.items() for v in values], "week int, x double"
    )
    out = weekly_psi(df, ref, {"x": [10.0]})
    return {r.week: r for r in out.collect()}


def test_bucket(spark):
    df = spark.createDataFrame([(v,) for v in [5.0, 10.0, 15.0, 20.0, 25.0, None]], "x double")
    out = [r.b for r in df.select(bucket("x", [10.0, 20.0]).alias("b")).collect()]

    assert out == [0, 1, 1, 2, 2, NULL_BUCKET]


def test_quantile_edges_drops_repeated_edges(spark):
    df = spark.createDataFrame([(v,) for v in [0.0] * 8 + [1.0, 2.0]], "x double")

    # Deciles 10% to 80% are all 0 and 90% is the 9th value, 1: nine edges collapse to two.
    assert quantile_edges(df, "x", n_buckets=10) == [0.0, 1.0]


def test_same_distribution_gives_zero(spark):
    out = psi_by_week(spark, [1.0, 1.0, 20.0, 20.0], {1: [5.0, 30.0]})

    assert out[1].psi == pytest.approx(0.0)
    assert out[1].rows == 2


def test_known_psi(spark):
    # Reference 50/50, week 90/10:
    # (0.9 - 0.5) ln(0.9 / 0.5) + (0.1 - 0.5) ln(0.1 / 0.5)
    expected = 0.4 * math.log(0.9 / 0.5) + (-0.4) * math.log(0.1 / 0.5)
    out = psi_by_week(spark, [1.0] * 5 + [20.0] * 5, {1: [1.0] * 9 + [20.0]})

    assert out[1].psi == pytest.approx(expected)


def test_empty_bucket_is_finite(spark):
    out = psi_by_week(spark, [1.0, 20.0], {1: [1.0, 2.0]})

    assert math.isfinite(out[1].psi)
    assert out[1].psi > 1.0


def test_new_nulls_count_as_drift(spark):
    out = psi_by_week(spark, [1.0, 20.0], {1: [1.0, 20.0], 2: [1.0, None]})

    assert out[1].psi == pytest.approx(0.0)
    assert out[2].psi > 1.0


def test_with_week(spark):
    df = spark.createDataFrame([(d,) for d in [0, 6, 7, 13, 14]], "event_day int")

    assert [r.week for r in with_week(df).collect()] == [0, 0, 1, 1, 2]
