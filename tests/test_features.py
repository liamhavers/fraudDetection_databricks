from fraud.features import add_frequency_features, add_velocity_features

COLS = "TransactionID int, card1 int, TransactionDT int, TransactionAmt double"


def velocity(spark, rows):
    df = spark.createDataFrame(rows, COLS)
    out = add_velocity_features(df, key="card1", prefix="card")
    return {r.TransactionID: r for r in out.collect()}


def test_counts_only_earlier_transactions_in_window(spark):
    out = velocity(spark, [(1, 7, 0, 10.0), (2, 7, 1800, 20.0), (3, 7, 5000, 30.0)])

    assert [out[i].card_txn_1h for i in (1, 2, 3)] == [0, 1, 1]
    assert [out[i].card_amt_1h for i in (1, 2, 3)] == [0.0, 10.0, 20.0]
    assert [out[i].card_txn_24h for i in (1, 2, 3)] == [0, 1, 2]
    assert [out[i].card_amt_24h for i in (1, 2, 3)] == [0.0, 10.0, 30.0]


def test_window_start_is_inclusive(spark):
    out = velocity(spark, [(1, 7, 0, 10.0), (2, 7, 3600, 20.0), (3, 7, 7201, 30.0)])

    assert out[2].card_txn_1h == 1  # 0 is exactly 3600s earlier, so inside
    assert out[3].card_txn_1h == 0  # 3600 is 3601s earlier, so outside


def test_same_second_transactions_do_not_see_each_other(spark):
    out = velocity(spark, [(1, 7, 100, 10.0), (2, 7, 100, 20.0)])

    assert out[1].card_txn_1h == 0
    assert out[2].card_txn_1h == 0
    assert out[1].card_secs_since_last is None


def test_keys_do_not_share_history(spark):
    out = velocity(spark, [(1, 7, 0, 10.0), (2, 8, 1800, 20.0)])

    assert out[2].card_txn_1h == 0


def test_null_key_gives_null_features(spark):
    out = velocity(spark, [(1, None, 0, 10.0), (2, None, 1800, 20.0)])

    for i in (1, 2):
        assert out[i].card_txn_1h is None
        assert out[i].card_amt_24h is None
        assert out[i].card_secs_since_last is None


def test_secs_since_last(spark):
    out = velocity(spark, [(1, 7, 0, 10.0), (2, 7, 1800, 20.0), (3, 7, 5000, 30.0)])

    assert [out[i].card_secs_since_last for i in (1, 2, 3)] == [None, 1800, 3200]


def test_frequency_counts_each_value(spark):
    df = spark.createDataFrame(
        [(1, "a", 5), (2, "a", 5), (3, "b", 6), (4, None, 5)],
        "TransactionID int, card4 string, card1 int",
    )
    out = add_frequency_features(df, cols=["card4", "card1"])
    rows = {r.TransactionID: r for r in out.collect()}

    assert [rows[i].card4_freq for i in (1, 2, 3, 4)] == [2, 2, 1, None]
    assert [rows[i].card1_freq for i in (1, 2, 3, 4)] == [3, 3, 1, 3]


def test_frequency_keeps_rows_and_column_order(spark):
    df = spark.createDataFrame(
        [(1, "a"), (2, "a"), (3, None)], "TransactionID int, card4 string"
    )
    out = add_frequency_features(df, cols=["card4"])

    assert out.count() == 3
    assert out.columns == ["TransactionID", "card4", "card4_freq"]
