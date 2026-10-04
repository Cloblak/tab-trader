"""Off-the-shelf momentum features for Kalshi BTC binary contracts.

Every feature is computed from information available at the decision instant
``decision_ts``:

* the contract's own top-of-book (now, 60 s ago, 180 s ago), and
* 1-minute BTC closes whose bar ENDED at or before ``decision_ts``.

All indicator settings are textbook defaults (RSI-14, Stochastic-14, CCI-20,
MACD 12/26/9, EMA 9/21, Bollinger 20/2, ...). Nothing here is tuned. The only
trend-line feature is a single 10-minute OLS slope and its R^2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

# Grouped for the write-up; FEATURES is the model input, in this order.
FEATURE_GROUPS: dict[str, list[str]] = {
    "contract": ["p_mid", "logit_mid", "spread_c", "tau_min", "d_mid_60s", "d_mid_180s"],
    "moneyness": ["log_moneyness_bps", "z_moneyness", "phi_z"],
    "btc_momentum": ["rsi_14", "stoch_k_14", "cci_20", "macd_hist_bps", "ema_gap_9_21_bps",
                     "bb_pctb_20", "zscore_60m", "range_pos_60m", "slope_10m_bps", "r2_10m"],
    "volatility": ["bb_width_20_bps", "rv_15m_bps", "rv_60m_bps"],
    "calendar": ["hour_sin", "hour_cos"],
}
FEATURES: list[str] = [f for g in FEATURE_GROUPS.values() for f in g]

# A bar older than this at the decision instant means the spot feed was down.
MAX_SPOT_AGE = pd.Timedelta(minutes=2)


def btc_grid(btc_1m: pd.DataFrame) -> pd.Series:
    """1-minute closes on a regular grid indexed by bar END (gaps forward-filled up to 2 bars)."""
    s = btc_1m.set_index("bar_end")["close"].sort_index()
    s.index = pd.DatetimeIndex(s.index).tz_convert("UTC").as_unit("ns")
    s = s[~s.index.duplicated(keep="last")]
    grid = pd.date_range(s.index[0], s.index[-1], freq="1min")
    return s.reindex(grid).ffill(limit=2)


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _ols_slope_r2(y: pd.Series, n: int) -> tuple[pd.Series, pd.Series]:
    """Rolling OLS of y on t = 0..n-1: slope per bar and R^2."""
    t = np.arange(n, dtype=float)
    tc = t - t.mean()
    stt = (tc ** 2).sum()

    def slope(w: np.ndarray) -> float:
        return float((tc * (w - w.mean())).sum() / stt)

    def r2(w: np.ndarray) -> float:
        yc = w - w.mean()
        syy = (yc ** 2).sum()
        if syy <= 1e-18:  # flat window: no trend, not a "perfect" one
            return 0.0
        return float((tc * yc).sum() ** 2 / (stt * syy))

    roll = y.rolling(n, min_periods=n)
    return roll.apply(slope, raw=True), roll.apply(r2, raw=True)


def btc_indicators(close: pd.Series) -> pd.DataFrame:
    """Indicator table on the 1-minute grid (index = bar end)."""
    lp = np.log(close)
    r = lp.diff()
    out = pd.DataFrame(index=close.index)
    out["btc"] = close
    ema = lambda n: close.ewm(span=n, adjust=False, min_periods=n).mean()  # noqa: E731
    macd = (ema(12) - ema(26)) / close * 1e4
    out["macd_hist_bps"] = macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()
    out["ema_gap_9_21_bps"] = (ema(9) - ema(21)) / close * 1e4
    out["rsi_14"] = _rsi(close, 14)
    lo14 = close.rolling(14, min_periods=14).min()
    hi14 = close.rolling(14, min_periods=14).max()
    out["stoch_k_14"] = (100 * (close - lo14) / (hi14 - lo14).replace(0, np.nan)).fillna(50.0)
    sma20 = close.rolling(20, min_periods=20).mean()
    mad20 = close.rolling(20, min_periods=20).apply(lambda w: np.abs(w - w.mean()).mean(), raw=True)
    out["cci_20"] = ((close - sma20) / (0.015 * mad20.replace(0, np.nan))).clip(-500, 500)
    sma = sma20
    sd = close.rolling(20, min_periods=20).std()
    out["bb_pctb_20"] = (close - (sma - 2 * sd)) / (4 * sd).replace(0, np.nan)
    out["bb_width_20_bps"] = 4 * sd / sma * 1e4
    m60 = close.rolling(60, min_periods=60).mean()
    s60 = close.rolling(60, min_periods=60).std()
    out["zscore_60m"] = ((close - m60) / s60.replace(0, np.nan)).clip(-6, 6)
    slope, r2 = _ols_slope_r2(lp, 10)
    out["slope_10m_bps"] = slope * 1e4
    out["r2_10m"] = r2
    hi = close.rolling(60, min_periods=60).max()
    lo = close.rolling(60, min_periods=60).min()
    out["range_pos_60m"] = ((close - lo) / (hi - lo).replace(0, np.nan)).fillna(0.5)
    out["rv_15m_bps"] = r.rolling(15, min_periods=15).std() * 1e4
    out["rv_60m_bps"] = r.rolling(60, min_periods=60).std() * 1e4
    # The bar that is "current" at each grid point actually exists (not forward-filled past 2 bars).
    out["spot_ok"] = close.notna()
    return out


def build_features(snap: pd.DataFrame, btc_1m: pd.DataFrame) -> pd.DataFrame:
    """Attach FEATURES to a snapshot table (one row per market x decision time)."""
    ind = btc_indicators(btc_grid(btc_1m))
    # Decisions sit on minute boundaries. Match units/tz explicitly: parquet
    # round-trips can turn ns into us, and a unit mismatch makes reindex miss silently.
    t = pd.DatetimeIndex(snap["decision_ts"].dt.floor("1min")).tz_convert("UTC").as_unit("ns")
    ind.index = pd.DatetimeIndex(ind.index).tz_convert("UTC").as_unit("ns")
    look = ind.reindex(t)
    look.index = snap.index
    df = pd.concat([snap, look], axis=1)

    mid = (df["bid"] + df["ask"]) / 2
    df["p_mid"] = mid / 100
    pm = df["p_mid"].clip(0.005, 0.995)
    df["logit_mid"] = np.log(pm / (1 - pm))
    df["spread_c"] = df["ask"] - df["bid"]
    df["tau_min"] = df["tau_s"] / 60
    df["d_mid_60s"] = mid - (df["bid_lag60"] + df["ask_lag60"]) / 2
    df["d_mid_180s"] = mid - (df["bid_lag180"] + df["ask_lag180"]) / 2

    lm = np.log(df["btc"] / df["strike"])
    df["log_moneyness_bps"] = lm * 1e4
    sig = (df["rv_60m_bps"] / 1e4) * np.sqrt(df["tau_min"])
    df["z_moneyness"] = (lm / sig.replace(0, np.nan)).clip(-8, 8)
    df["phi_z"] = ndtr(df["z_moneyness"])

    h = df["decision_ts"].dt.hour + df["decision_ts"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    df["spot_ok"] = df["spot_ok"].astype("boolean").fillna(False).astype(bool)
    return df
