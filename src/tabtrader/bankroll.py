"""$100 bankroll simulation and a small tear sheet.

Trades are taken one at a time in order (day, then ``seq`` within the day). Each trade buys
contracts = min(stake fraction × current balance / entry price, MAX_CONTRACTS): 15% of the balance
(or a per-trade half-Kelly fraction), never more than 500 contracts. The cap stands in for the depth of
Kalshi's order book; without it 15% stakes compound to sizes no one could fill. Contracts are fractional
below the cap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

START = 100.0
RISK = 0.15
MAX_CONTRACTS = 500
ANN = 365  # crypto trades every day


def daily_equity(day: pd.Series, net_c: np.ndarray, px_c: np.ndarray, take: np.ndarray,
                 days: pd.DatetimeIndex | None = None, start: float = START, risk=RISK,
                 max_contracts: float | None = MAX_CONTRACTS, seq=None) -> pd.Series:
    """Account value at the end of each day. ``risk`` is a fraction, or one fraction per trade."""
    n = len(net_c)
    f = pd.DataFrame({"day": pd.to_datetime(pd.Series(day).to_numpy()), "net": np.asarray(net_c, float),
                      "px": np.asarray(px_c, float), "take": np.asarray(take, bool),
                      "frac": np.broadcast_to(np.asarray(risk, float), (n,)),
                      "seq": np.arange(n) if seq is None else np.asarray(seq)})
    f = f.sort_values(["day", "seq"], kind="stable")
    eq, end = start, {}
    for d_, net, px, tk, fr in zip(f["day"], f["net"], f["px"], f["take"], f["frac"]):
        if tk and fr > 0 and eq > 0:
            contracts = fr * eq / (px / 100)
            if max_contracts is not None:
                contracts = min(contracts, max_contracts)
            eq += contracts * net / 100
        end[d_] = eq
    out = pd.Series(end).sort_index()
    if days is not None:
        out = out.reindex(days).ffill().fillna(start)
    return out


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
        seq = d["seq"].to_numpy() if "seq" in d else None
        eq = daily_equity(d["day"], d[net_col].to_numpy(), d["px"].to_numpy(), take, days, risk=frac, seq=seq)
        ts = tear_sheet(eq, int(take.sum()), int((d[net_col].to_numpy()[take] > 0).sum()))
        ts["curve"] = [[str(k.date()), round(float(v), 2)] for k, v in eq.items()]
        ts["c_per_ct"] = round(float(d[net_col].to_numpy()[take].mean()), 2) if take.any() else None
        if take.sum() >= 10:  # day-block 95% range of the mean (trade-weighted)
            x = pd.Series(d[net_col].to_numpy()[take], index=d["day"].to_numpy()[take])
            g = x.groupby(level=0).agg(["sum", "count"])
            rng = np.random.default_rng(0)
            idx = rng.integers(0, len(g), size=(2000, len(g)))
            bs = g["sum"].to_numpy()[idx].sum(1) / g["count"].to_numpy()[idx].sum(1)
            ts["c_ci95"] = [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]
        out[m] = ts
    return out


def boot_final_ratio(df: pd.DataFrame, a: str, b: str, reps: int = 1000, seed: int = 0,
                     net_col: str = "net_c_exits") -> dict:
    """95% range of final(a) / final(b) by resampling whole days (same draw for both), capped simulation."""
    d = df.copy()
    d["_day"] = pd.to_datetime(d["day"])
    d["_seq"] = d["seq"] if "seq" in d else d.groupby("_day").cumcount()
    days = d["_day"].drop_duplicates().sort_values().to_numpy()
    groups = {k: g for k, g in d.groupby("_day")}
    rng = np.random.default_rng(seed)

    def final(rows: pd.DataFrame, m: str) -> float:
        frac = rows[f"frac__{m}"].to_numpy(float) if f"frac__{m}" in rows else RISK
        eq = daily_equity(rows["_bday"], rows[net_col].to_numpy(), rows["px"].to_numpy(),
                          rows[f"take__{m}"].to_numpy(bool), risk=frac, seq=rows["_seq"].to_numpy())
        return float(eq.iloc[-1])

    ratios = []
    for _ in range(reps):
        pick = rng.choice(len(days), size=len(days), replace=True)
        rows = pd.concat([groups[days[i]].assign(_bday=pd.Timestamp("2000-01-01") + pd.Timedelta(days=j))
                          for j, i in enumerate(pick)], ignore_index=True)
        ratios.append(final(rows, a) / max(final(rows, b), 1e-9))
    r = np.array(ratios)
    return {"lo": float(np.percentile(r, 2.5)), "hi": float(np.percentile(r, 97.5)),
            "p_a_better": float(np.mean(r > 1))}
