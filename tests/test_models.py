def test_isolation_forest(tiny_df):
    from fraud_unsup.models.isolation_forest import IFDetector
    num = tiny_df.select_dtypes("number")
    model = IFDetector().fit(num)
    scores = model.score(num)
    assert scores.shape[0] == len(tiny_df)
