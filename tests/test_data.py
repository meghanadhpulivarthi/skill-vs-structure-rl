import numpy as np
import pandas as pd
from src.data import compute_returns, load_etf_panel


def test_compute_returns_drops_first_row_and_shapes():
    prices = pd.DataFrame(
        {"A": [100.0, 101.0, 99.0], "B": [50.0, 50.0, 55.0]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
    )
    rets = compute_returns(prices)
    assert list(rets.columns) == ["A", "B"]
    assert rets.shape == (2, 2)                       # first row (NaN) dropped
    np.testing.assert_allclose(rets["A"].iloc[0], 0.01, atol=1e-9)
    np.testing.assert_allclose(rets["B"].iloc[1], 0.10, atol=1e-9)


def test_compute_returns_drops_ticker_with_gap():
    prices = pd.DataFrame(
        {"A": [100.0, 101.0, 102.0], "B": [50.0, np.nan, 55.0]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
    )
    rets = compute_returns(prices)
    assert list(rets.columns) == ["A"]                # B dropped for the NaN gap


def test_load_etf_panel_uses_cache(tmp_path):
    # A pre-existing cache must be read without any network download.
    cache = tmp_path / "panel.parquet"
    prices = pd.DataFrame(
        {"A": [100.0, 101.0, 102.0, 103.0], "B": [50.0, 51.0, 50.0, 52.0]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
    )
    prices.to_parquet(cache)
    panel = load_etf_panel(cache_path=cache)
    assert panel["returns"].shape == (3, 2)           # 4 prices -> 3 returns
    assert panel["tickers"] == ["A", "B"]
    assert panel["dates"].shape == (3,)
