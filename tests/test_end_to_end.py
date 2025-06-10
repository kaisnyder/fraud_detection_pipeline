def test_end_to_end(tmp_path, monkeypatch, tiny_df):
    """Smoke-test: train → save → load → score."""
    import pandas as pd
    from fraud_unsup.models.isolation_forest import IFDetector
    path = tmp_path / "df.parquet"
    tiny_df.to_parquet(path)
    num = tiny_df.select_dtypes("number")
    mdl = IFDetector(n_estimators=10).fit(num)
    mdl.save(tmp_path / "iso.joblib")
    mdl2 = IFDetector.load(tmp_path / "iso.joblib")
    assert (mdl.score(num) == mdl2.score(num)).all()
