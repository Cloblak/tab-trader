"""Load the published snapshot tables, apply the cleaning rules and attach features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .features import FEATURES, build_features

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


@dataclass(frozen=True)
class MarketSpec:
    key: str
    title: str
    snapshot_file: str
    first_test_block: str  # Monday, UTC
    n_blocks: int
    n_val: int
    mid_range: tuple[float, float] | None  # cents; None = no filter


MARKETS = {
    "15m": MarketSpec("15m", "KXBTC15M · 15-minute up/down", "kxbtc15m.parquet",
                      "2026-08-03", 9, 1250, None),
    "1h": MarketSpec("1h", "KXBTCD · hourly strike ladder", "kxbtcd.parquet",
                     "2026-08-24", 6, 750, (10.0, 90.0)),
}
MAX_CANDLE_DIFF_C = 2.0


def clean_mask(df: pd.DataFrame, spec: MarketSpec) -> pd.Series:
    """The cleaning rule, as one boolean mask (see notebook 03)."""
    m = (df["candle_status"] == "ok") & (df["candle_close_diff_c"] <= MAX_CANDLE_DIFF_C)
    if "spot_ok" in df:
        m &= df["spot_ok"]
    if spec.mid_range is not None:
        mid = (df["bid"] + df["ask"]) / 2
        m &= mid.between(*spec.mid_range)
    return m


def load_market(key: str, clean: bool = True) -> pd.DataFrame:
    spec = MARKETS[key]
    snap = pd.read_parquet(DATA / "snapshots" / spec.snapshot_file)
    btc = pd.read_parquet(DATA / "btc_1m.parquet")
    df = build_features(snap, btc)
    if clean:
        df = df[clean_mask(df, spec)]
    df = df.sort_values(["decision_ts", "market"]).reset_index(drop=True)
    df["row_id"] = df["market"] + "|" + df["tau_s"].astype(str)
    df["day"] = df["decision_ts"].dt.floor("D")
    return df


def blocks(spec: MarketSpec) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(spec.first_test_block, tz="UTC")
    out = []
    for i in range(spec.n_blocks):
        a = start + pd.Timedelta(days=7 * i)
        b = a + pd.Timedelta(days=7)
        out.append((a, min(b, pd.Timestamp("2026-10-04", tz="UTC"))))
    return out


def split_block(df: pd.DataFrame, spec: MarketSpec, a: pd.Timestamp, b: pd.Timestamp,
                n_fit: int | None):
    """Return (fit, val, test) frames for test block [a, b).

    The pool is every row whose market closed before ``a``; validation is its
    most recent ``spec.n_val`` rows; the fit set is the ``n_fit`` rows before
    that (all of them when ``n_fit`` is None).
    """
    pool = df[df["close_time"] < a]
    val = pool.iloc[-spec.n_val:]
    before = pool.iloc[: len(pool) - spec.n_val]
    fit = before if n_fit is None else before.iloc[-n_fit:]
    test = df[(df["decision_ts"] >= a) & (df["decision_ts"] < b)]
    assert fit["close_time"].max() < a and val["close_time"].max() < a
    assert len(fit) == 0 or fit["decision_ts"].max() <= val["decision_ts"].min()
    return fit, val, test


def xy(df: pd.DataFrame):
    return df[FEATURES].to_numpy(float), df["label"].to_numpy(int)
