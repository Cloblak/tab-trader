"""Repository checks: leakage, reproducibility, claims, controls, privacy, size.

    python scripts/verify.py          # exits non-zero if any check fails

Private-string checks read optional extra patterns from the environment
(TT_PRIVATE_PATTERNS, '|'-separated) so the private values themselves never enter the repo.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tabtrader.data import MARKETS, blocks, load_market, split_block  # noqa: E402
from tabtrader.evaluate import load_meta, load_preds  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))


def leakage() -> None:
    for mk, spec in MARKETS.items():
        df = load_market(mk)
        close = df.set_index("row_id")["close_time"]
        bad = 0
        for i, (a, b) in enumerate(blocks(spec)):
            fit, val, test = split_block(df, spec, a, b, None)  # asserts inside
            for model in ["TabPFN-3.5", "LightGBM · tuned", "Market price"]:
                pr = load_preds(mk, "n5000", model)
                if pr is None:
                    continue
                v = pr[(pr.block == i) & (pr.split == "val")]
                bad += int((close.loc[v.row_id] >= a).sum())
                t = pr[(pr.block == i) & (pr.split == "test")]
                tt = df.set_index("row_id").loc[t.row_id, "decision_ts"]
                bad += int(((tt < a) | (tt >= b)).sum())
            for m in load_meta(mk, "n5000", "TabPFN-3.5"):
                if m["block"] == i:
                    bad += int(pd.Timestamp(m["fit_to"]) >= a) + int(pd.Timestamp(m["val_to"]) >= a)
        check(f"no train/val row from the test week or later ({mk})", bad == 0, f"{bad} violations")


def reproducibility() -> None:
    q = pd.read_parquet(ROOT / "data/raw_week/kxbtc15m_quotes_1s.parquet")
    pub = pd.read_parquet(ROOT / "data/btc_1m.parquet").set_index("bar_end").close
    sec = q.groupby("sec_end").btc.last().dropna()
    reb = sec[sec.index.second == 0]
    same = np.isclose(pub.reindex(reb.index), reb).mean()
    check("BTC 1-minute bars rebuild from the raw week", same > 0.999, f"{same:.4%} identical")
    snap = pd.read_parquet(ROOT / "data/snapshots/kxbtc15m.parquet")
    wk = snap[(snap.decision_ts >= "2026-09-26") & (snap.decision_ts < "2026-10-03") & snap.bid.notna()]
    qq = q[q.quote_valid == 1].rename(columns={"sec_end": "decision_ts"})
    m = wk.merge(qq[["market", "decision_ts", "bid", "ask"]], on=["market", "decision_ts"], suffixes=("", "_raw"))
    rate = ((m.bid == m.bid_raw) & (m.ask == m.ask_raw)).mean()
    check("decision quotes rebuild from the raw week", rate > 0.9, f"{rate:.1%} identical of {len(m)}")
    a = load_market("15m")
    b = load_market("15m")
    from tabtrader.features import FEATURES
    check("features are deterministic", a[FEATURES].equals(b[FEATURES]))


def claims() -> None:
    S = json.loads((ROOT / "results/summary.json").read_text())
    html = (ROOT / "docs/index.html").read_text()
    for mk in MARKETS:
        L = S[mk]["policy_A"]["models"]
        if "TabPFN-3.5" in L:
            v = f"{L['TabPFN-3.5']['logloss']:.4f}"
            check(f"report quotes TabPFN-3.5 log loss from results ({mk})", v in html, v)
        emb = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S).group(1)
        D = json.loads(emb)
        check(f"report embeds the current summary ({mk})",
              D["bench"][mk]["policy_A"]["n_rows"] == S[mk]["policy_A"]["n_rows"])
    for mk in MARKETS:
        L = S[mk]["policy_A"]["models"]
        n = {m: r["n"] for m, r in L.items()}
        check(f"every model scored on identical test rows ({mk})", len(set(n.values())) == 1, str(set(n.values())))


def controls() -> None:
    p = ROOT / "results/controls.json"
    if not p.exists():
        check("controls ran", False, "results/controls.json missing")
        return
    c = json.loads(p.read_text())
    for m in ("TabPFN-3.5", "Logistic regression · default"):
        v = c["planted"][m]["minus_market"]
        check(f"planted signal detected by {m}", v["hi"] < 0, f"{v}")
    missed = [m for m, v in c["planted"].items() if v["minus_market"] and v["minus_market"]["hi"] >= 0]
    if missed:
        print(f"note: planted signal not significant for {missed} (reported on the page)")
    for m, v in c["null"].items():
        if v["minus_market"]:
            check(f"no false skill under the calibrated null: {m}", v["minus_market"]["lo"] > -0.002 or v["minus_market"]["hi"] >= 0,
                  f"{v['minus_market']}")


def privacy() -> None:
    files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True,
                           text=True).stdout.split()
    pats = [r"192\.168\.\d+\.\d+", r"BEGIN (RSA |EC )?PRIVATE KEY", r"TABPFN_TOKEN\s*=\s*\S{12,}",
            r"CLICKHOUSE_PASSWORD\s*=\s*\S+", r"\.pem\b"]
    extra = os.environ.get("TT_PRIVATE_PATTERNS", "")
    pats += [r"(?<![A-Za-z0-9_+/])" + re.escape(x) + r"(?![A-Za-z0-9_+/])" for x in extra.split("|") if x]
    tok = os.environ.get("TABPFN_TOKEN")
    if tok:
        pats.append(re.escape(tok))
    hits = []
    for f in files:
        path = ROOT / f
        if path.suffix in (".parquet", ".png") or not path.is_file() or f == "scripts/verify.py":
            continue
        if f == "uv.lock":
            continue
        txt = path.read_text(errors="ignore")
        if path.suffix == ".ipynb":  # scan text and code, not embedded images
            nb = json.loads(txt)
            parts = []
            for c in nb["cells"]:
                parts.append("".join(c.get("source", "")))
                for o in c.get("outputs", []):
                    parts.append("".join(o.get("text", "")))
                    parts.append("".join(o.get("data", {}).get("text/plain", "")))
            txt = "\n".join(parts)
        for p in pats:
            if re.search(p, txt):
                hits.append(f"{f}: /{p[:30]}/")
    check("no secrets, hosts or private strings in tracked files", not hits, "; ".join(hits[:5]))
    for f, pat in [("rolling_predictions", r"^(day|week|split|px|won|net_c_exits|take__.*|p__.*|frac__.*)$"),
                   ("rolling_live_trades", r"^(day|px_fill|real_net_c|tabpfn_keeps|frac_kelly)$"),
                   ("bluf_predictions", r"^(day|split|px|won|net_c_hold|net_c_exits|take__.*|p__.*|frac__.*)$"),
                   ("live_trades", r"^(day|px_fill|real_net_c|tabpfn_keeps|backtest_live_gate_takes|frac_kelly)$")]:
        d = pd.read_parquet(ROOT / f"data/strategy/{f}.parquet")
        allowed = re.compile(pat)
        check(f"{f}: only scores, decisions and outcomes (no features)", all(allowed.match(c) for c in d.columns),
              ",".join(c for c in d.columns if not allowed.match(c)))
        check(f"{f}: no intraday timing", d.day.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all())


def sizes() -> None:
    files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True,
                           text=True).stdout.split()
    sz = {f: (ROOT / f).stat().st_size for f in files if (ROOT / f).is_file()}
    big = {f: s for f, s in sz.items() if s > 50e6}
    check("no file over 50 MB", not big, str(big))
    check("repository under 150 MB", sum(sz.values()) < 150e6, f"{sum(sz.values())/1e6:.1f} MB")


def main() -> None:
    for fn in (leakage, reproducibility, claims, controls, privacy, sizes):
        try:
            fn()
        except Exception as e:  # a crashed check is a failed check
            check(fn.__name__, False, f"{type(e).__name__}: {e}")
    w = max(len(n) for n, _, _ in RESULTS)
    for n, ok, d in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {n:<{w}}  {d}")
    failed = sum(not ok for _, ok, _ in RESULTS)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} checks pass")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
