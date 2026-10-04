"""Write the results block of README.md from results/summary.json (numbers are never typed by hand)."""

from __future__ import annotations

import json
import re

from .data import MARKETS, ROOT


def block() -> str:
    S = json.loads((ROOT / "results" / "summary.json").read_text())
    out = []
    for mk in MARKETS:
        L = S[mk]["policy_A"]
        rows = sorted(L["models"].items(), key=lambda kv: kv[1]["logloss"])
        out.append(f"**{MARKETS[mk].title}** · {L['n_rows']:,} test rows over {L['n_days']} days "
                   f"(policy A: every model sees the same 5,000 rows)\n")
        out.append("| Model | Log loss | Δ vs TabPFN-3.5 [95% CI] | Δ vs market | AUC | ECE | CPU s / refit |")
        out.append("|---|---|---|---|---|---|---|")
        for m, r in rows:
            v, mkv = r.get("vs_ref"), r.get("vs_market")
            cpu = r.get("cpu", {})
            t = (cpu.get("fit_s_mean") or 0) + (cpu.get("search_s_mean") or 0)
            name = f"**{m}**" if m.startswith("TabPFN") else m
            out.append(f"| {name} | {r['logloss']:.4f} | "
                       + (f"{v['mean']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]{' ✱' if v.get('bh_pass') else ''}" if v else "reference")
                       + f" | {'' if not mkv else f'{mkv['mean']:+.4f}'} | {r['auc']:.4f} | {r['ece']:.4f} | "
                       + ("–" if m == "Market price" else f"{t:.1f}") + " |")
        out.append("")
    out.append("✱ significant after Benjamini–Hochberg (q = 0.10). Δ > 0 means worse than the reference. "
               "CPU seconds include the hyper-parameter search for classic models; TabPFN is never tuned.\n")
    return "\n".join(out)


def main() -> None:
    p = ROOT / "README.md"
    s = p.read_text()
    new = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
                 lambda _: "<!-- RESULTS:START -->\n" + block() + "\n<!-- RESULTS:END -->", s, flags=re.S)
    p.write_text(new)
    print("README results block updated")


if __name__ == "__main__":
    main()
