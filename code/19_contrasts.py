"""SPEC v1 §1 pre-registered contrasts C1-C4 (Product, main split), month-block bootstrap
(blocks of 3 consecutive months, resampled within each window, 2,000 replicates), two-sided
p-values, Holm correction across the available contrasts.

Statistic: difference in excess errors per 1k (policy X minus policy Y), pooled over windows A and B
as sum(excess)/sum(n)*1000 on the resampled months. C3 and C4 also report the automation difference.
  C1 rolling_cod_known vs static_plugin          all eval months
  C2 rolling_cod_known vs rolling_plugin_known   all eval months
  C3 rolling_cod_known vs rolling_cod_none       break .. break+6
  C4 C4_with_llm vs C4_without_llm               break .. break+6 (automation is the headline)

Usage: python -I scripts/19_contrasts.py <results_dir> [alpha]
Output: results/v1/contrasts_alpha<a>.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

res = Path(sys.argv[1]); alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
R = pd.read_csv(res / "v1" / "main_monthly.csv")
R = R[(R["task"] == "product") & (R["split"] == "main") & (R["alpha"] == alpha)]
BRK = {"A": "2017-04", "B": "2023-08"}
rng = np.random.default_rng(20261102)


def mshift(m, k):
    return str(pd.Period(m, "M") + k)


def window_months(per, which):
    b = BRK[per]
    if which == "all":
        return None
    if which == "post6":
        return [mshift(b, k) for k in range(0, 7)]
    return [mshift(b, k) for k in range(-2, 7)]


def paired(px, py, which):
    rows = []
    for per in ["A", "B"]:
        x, y = R[(R["policy"] == px) & (R["period"] == per)], R[(R["policy"] == py) & (R["period"] == per)]
        j = x.merge(y, on="month", suffixes=("_x", "_y"))
        wm = window_months(per, which)
        if wm is not None:
            j = j[j["month"].isin(wm)]
        rows.append(j.assign(period=per).sort_values("month"))
    return pd.concat(rows, ignore_index=True)


def stat(j):
    ex = 1000 * (j["excess_x"].sum() / j["n_x"].sum() - j["excess_y"].sum() / j["n_y"].sum())
    au = j["automation_x"].mean() - j["automation_y"].mean()
    return ex, au


def boot(j, B=2000, block=3):
    out = []
    for _ in range(B):
        parts = []
        for per, g in j.groupby("period"):
            g = g.reset_index(drop=True)
            nb = int(np.ceil(len(g) / block))
            starts = rng.integers(0, max(1, len(g) - block + 1), nb)
            idx = np.concatenate([np.arange(s, min(s + block, len(g))) for s in starts])[:len(g)]
            parts.append(g.iloc[idx])
        out.append(stat(pd.concat(parts)))
    return np.array(out)


def p2(d):
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


specs = [("C1", "rolling_cod_known", "static_plugin", "all"),
         ("C2", "rolling_cod_known", "rolling_plugin_known", "all"),
         ("C3", "rolling_cod_known", "rolling_cod_none", "post6"),
         ("C4", "C4_with_llm", "C4_without_llm", "post6")]  # SPEC v1: C4 on break..break+6
rows = []
for name, px, py, which in specs:
    if not ((R["policy"] == px).any() and (R["policy"] == py).any()):
        continue
    j = paired(px, py, which)
    ex, au = stat(j)
    bs = boot(j)
    rows.append({"contrast": name, "x": px, "y": py, "months": which, "n_months": len(j),
                 "excess_per_1k_diff": round(ex, 3), "excess_ci95": [round(np.quantile(bs[:, 0], q), 3) for q in (0.025, 0.975)],
                 "p_excess": p2(bs[:, 0]), "automation_diff": round(au, 4),
                 "automation_ci95": [round(np.quantile(bs[:, 1], q), 4) for q in (0.025, 0.975)], "p_automation": p2(bs[:, 1])})
T = pd.DataFrame(rows)
# Holm over the headline p of each contrast (C1-C3: excess; C4: automation)
head = np.where(T["contrast"] == "C4", T["p_automation"], T["p_excess"])
order = np.argsort(head); m = len(head); adj = np.empty(m); run = 0.0
for k, i in enumerate(order):
    run = max(run, min(1.0, (m - k) * head[i])); adj[i] = run
T["p_headline"], T["p_holm"] = head, adj
T.to_csv(res / "v1" / f"contrasts_alpha{alpha:.2f}.csv", index=False)
pd.set_option("display.width", 250)
print(T.to_string(index=False))
