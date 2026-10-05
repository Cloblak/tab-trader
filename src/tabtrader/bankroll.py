"""$100 bankroll simulation and a small tear sheet.

Every trade stakes ``RISK`` (15%) of the current account on the contract: stake = 0.15 × equity,
contracts = stake / entry price, so a trade's account return is 0.15 × net¢ / entry¢. Stakes are
fractional, which keeps each day's growth factor independent of the order of trades within the day.
That is why the published files only need a date per trade. (Whole contracts change results by
well under 1% at these sizes.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

START = 100.0
RISK = 0.15
ANN = 365  # crypto trades every day


def daily_equity(day: pd.Series, net_c: np.ndarray, px_c: np.ndarray, take: np.ndarray,
                 days: pd.DatetimeIndex | None = None, start: float = START, risk=RISK) -> pd.Series:
    """Account value at the end of each day. ``risk`` is a fraction, or one fraction per trade."""
    r = np.log1p(np.asarray(risk, float) * np.asarray(net_c, float) / np.asarray(px_c, float))
    g = pd.Series(np.where(take, r, 0.0), index=pd.to_datetime(day)).groupby(level=0).sum()
    if days is not None:
        g = g.reindex(days, fill_value=0.0)
    return start * np.exp(g.cumsum())


def tear_sheet(eq: pd.Series, n_trades: int, wins: int, start: float = START) -> dict:
    """Return, Sharpe, Sortino, max drawdown, win rate, and the weekly returns behind them."""
    eq0 = pd.concat([pd.Series([start], index=[eq.index[0] - pd.Timedelta(days=1)]), eq])
    dr = eq0.pct_change().dropna()
    sd = dr.std(ddof=1)
    down = np.sqrt((np.minimum(dr, 0) ** 2).mean())
    wk = eq0.resample("W-SUN").last()
    wk_ret = wk.pct_change().dropna()
    if len(wk) and len(eq0):  # first (partial) week measured from the start value
        first = wk.iloc[0] / start - 1
        wk_ret = pd.concat([pd.Series([first], index=[wk.index[0]]), wk_ret])
    peak = eq0.cummax()
    return {
        "final": round(float(eq.iloc[-1]), 2),
        "return_pct": round(100 * (eq.iloc[-1] / start - 1), 1),
        "sharpe": round(float(dr.mean() / sd * np.sqrt(ANN)), 2) if sd > 0 else None,
        "sortino": round(float(dr.mean() / down * np.sqrt(ANN)), 2) if down > 0 else None,
        "max_dd_pct": round(float(100 * (eq0 / peak - 1).min()), 1),
        "trades": int(n_trades),
        "win_rate": round(wins / n_trades, 3) if n_trades else None,
        "weeks_up": f"{int((wk_ret > 0).sum())}/{len(wk_ret)}",
        "weekly": [[str(k.date()), round(float(100 * v), 2)] for k, v in wk_ret.items()],
    }


def run_models(df: pd.DataFrame, models: list[str], split: str | None = None,
               net_col: str = "net_c_exits") -> dict:
    """Equity curve + tear sheet for each model's ``take__`` column on one split."""
    d = df if split is None else df[df["split"] == split]
    days = pd.date_range(pd.to_datetime(d["day"]).min(), pd.to_datetime(d["day"]).max(), freq="D")
    out = {}
    for m in models:
        take = d[f"take__{m}"].to_numpy(bool)
        frac = d[f"frac__{m}"].to_numpy(float) if f"frac__{m}" in d else RISK
        eq = daily_equity(d["day"], d[net_col].to_numpy(), d["px"].to_numpy(), take, days, risk=frac)
        ts = tear_sheet(eq, int(take.sum()), int((d[net_col].to_numpy()[take] > 0).sum()))
        ts["curve"] = [[str(k.date()), round(float(v), 2)] for k, v in eq.items()]
        ts["c_per_ct"] = round(float(d[net_col].to_numpy()[take].mean()), 2) if take.any() else None
        out[m] = ts
    return out
