"""Benchmark statistics table and timeline figure for the paper (from data files only).
Usage: python -I scripts/22_bench_stats.py <data_dir> <results_dir> <paper_dir>
Outputs: paper/tables/bench_stats.tex, figures/fig_timeline.pdf|png
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from labels import LABELS, NEW_CLASSES

data, res, paper = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
S = pd.read_parquet(data / "samples_v1.parquet", columns=["period", "set", "month", "regime", "product", "issue", "group"])
L = pd.read_parquet(data / "label_counts_narr.parquet").groupby("month")["n"].sum()
W = pd.read_parquet(data / "window_items_v1.parquet", columns=["date", "group_size"])
W["month"] = W["date"].str[:7]
PER = {"A": ("2016-04", "2018-12", "2017-04"), "D": ("2019-01", "2021-07", None), "B": ("2022-08", "2025-02", "2023-08")}
rows = []
for per, (a, b, brk) in PER.items():
    s = S[S["period"] == per]
    ev = s[s["set"] == "eval"]
    vol = L[(L.index >= a) & (L.index <= b)]
    w = W[(W["month"] >= a) & (W["month"] <= b)]
    regs = sorted(ev["regime"].unique())
    post = ev[ev["month"] > brk] if brk else ev.iloc[0:0]
    newsh = 100 * post["product"].isin(NEW_CLASSES[post["regime"].iloc[0]]).mean() if len(post) else np.nan
    cr = 100 * ev["product"].str.startswith("Credit reporting").mean()
    rows.append([per, f"{a}..{b}", f"{len(vol)}", f"{int(vol.sum()):,}", f"{int(vol.min()):,}--{int(vol.max()):,}",
                 "/".join(str(len(LABELS[r])) for r in regs), "--" if np.isnan(newsh) else f"{newsh:.1f}",
                 f"{cr:.0f}", f"{100*(w['group_size'] > 1).mean():.0f}", f"{len(ev):,}", f"{(s['set'].str.startswith('audit')).sum():,}"])
hdr = ["Period", "Eval months", "Months", "Narratives", "Per month", "Products", "New-class \\%", "Credit rep. \\%", "Near-dup \\%", "Stream", "Audit"]
body = "\n".join(" & ".join(r) + r" \\" for r in rows)
(paper / "tables" / "bench_stats.tex").write_text(r"""\begin{table*}[t]
\caption{Benchmark periods. Narratives: all CFPB narratives received in the eval months (per month: minimum--maximum). Products: label-set size before/after the break. New-class \%%: share of post-break stream complaints in classes created at the break. Credit rep.\ \%%: share of the stream in the credit-reporting product. Near-dup \%%: share of the sampled pool (at most 20{,}000 complaints per month) in MinHash groups (Jaccard $\geq 0.8$) of size $>1$. Stream and Audit: sampled complaints (2{,}000 and 600 per month).}
\label{tab:bench}
\small
\begin{tabular}{llrrrcrrrrr}
\toprule
%s \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % (" & ".join(hdr), body))
print(body)

# timeline figure
fig, ax = plt.subplots(figsize=(7.0, 1.35))
y = {"A": 2, "D": 1, "B": 0}
col = {"A": "#2166ac", "D": "#999999", "B": "#b2182b"}
for per, (a, b, brk) in PER.items():
    t0, t1 = pd.Timestamp(a + "-01"), pd.Timestamp(b + "-28")
    tr0 = pd.Timestamp(str(pd.Period(a, "M") - 12) + "-01")
    ax.barh(y[per], (t0 - tr0).days, left=tr0, height=0.5, color=col[per], alpha=0.25)
    ax.barh(y[per], (t1 - t0).days, left=t0, height=0.5, color=col[per], alpha=0.75)
    ax.text(tr0 - pd.Timedelta(days=40), y[per], {"A": "A", "D": "D (dev)", "B": "B"}[per], ha="right", va="center", fontsize=7)
for d, lab in [("2017-04-24", "break 1"), ("2023-08-25", "break 2")]:
    ax.axvline(pd.Timestamp(d), color="k", lw=0.8); ax.text(pd.Timestamp(d), 2.55, lab, fontsize=6, ha="center")
for d in ["2019-10-01", "2020-10-01"]:
    ax.axvline(pd.Timestamp(d), color="k", lw=0.6, ls=":"); ax.text(pd.Timestamp(d), 2.55, "placebo", fontsize=6, ha="center")
ax.axvline(pd.Timestamp("2026-08-14"), color="grey", lw=0.6, ls="--"); ax.text(pd.Timestamp("2026-08-14"), 2.55, "archive ends", fontsize=6, ha="center")
ax.set_yticks([]); ax.set_ylim(-0.5, 2.9); ax.tick_params(axis="x", labelsize=7)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
FIG = res.parent / "figures"
fig.savefig(FIG / "fig_timeline.pdf"); fig.savefig(FIG / "fig_timeline.png", dpi=150)
print("timeline written")
