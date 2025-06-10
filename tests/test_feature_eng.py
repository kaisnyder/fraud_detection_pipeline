def test_time_features(tiny_df):
    from fraud_unsup.features.temporal import add_time_features
    out = add_time_features(tiny_df)
    assert {"hour", "dow", "month"}.issubset(out.columns)
