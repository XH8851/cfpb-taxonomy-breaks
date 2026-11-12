"""Secondary analyses from existing outputs (no new model runs):
 1. Margin selection for plug-in and rolling CRC on the development period D with the SAME rule
    used for CoD's knobs (target 0.10; nominal level a' in {0.05, 0.10, 0.15}; maximize automation
    subject to excess <= 1 per 1k overall and within +-3 months of each placebo), then A/B scored
    against the 10% target, with a 3-month block bootstrap for CoD minus the D-selected rule.
 2. Within-window pre/post-break validity of static and rolling CoD (product, alpha = 0.10).
 3. Class-prior shift at each break: total-variation distance between the stream's product mix in
    the 3 months before and after the break (old labels mapped through the crosswalk; no-carry-over
    labels kept as their own category), and per-tier accuracy before/after.
Usage: python -I scripts/26_secondary.py <data_dir> <results_dir> <paper_dir>
Outputs: results/v1/secondary.json, paper/tables/dselected.tex, paper/tables/app_shift.tex, paper/numbers_secondary.tex
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from labels import map_label

data, res, paper = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
V = res / "v1"; T = paper / "tables"
R = pd.read_csv(V / "main_monthly.csv"); R = R[(R["task"] == "product") & (R["split"] == "main")]
DM = pd.read_csv(V / "dev_factorial_monthly.csv")
BRK = {"A": "2017-04", "B": "2023-08"}; ms = lambda m, k: str(pd.Period(m, "M") + k)
PLAC = ["2019-10", "2020-10"]; TARGET = 0.10
rng = np.random.default_rng(20261104)
out, num = {}, {}


def score(g, target=TARGET):
    return 1000 * np.maximum(0, g["n_err"] - target * g["n_auto"]).sum() / g["n"].sum(), 100 * g["automation"].mean()


# 1. margin selected on D
sel = {}
for pol in ["rolling_plugin_known", "crc"]:
    cands = []
    for a in [0.05, 0.10, 0.15]:
        g = DM[(DM["policy"] == pol) & (DM["alpha"] == a)]
        ex, au = score(g)
        near = max(score(g[(g["month"] >= ms(p, -3)) & (g["month"] <= ms(p, 3))])[0] for p in PLAC)
        cands.append((a, ex, near, au))
    ok = [c for c in cands if c[1] <= 1.0 and c[2] <= 1.0]
    sel[pol] = max(ok, key=lambda c: c[3])[0] if ok else min(cands, key=lambda c: c[1])[0]
out["d_selected_level"] = sel


def boot(x, y, block=3, B=2000):
    """CoD (x) minus rule (y): difference in excess per 1k (vs 10%) and automation (pts)."""
    j = x.merge(y, on=["period", "month"], suffixes=("_x", "_y"))
    def stat(d):
        ex = 1000 * (np.maximum(0, d["n_err_x"] - TARGET * d["n_auto_x"]).sum() - np.maximum(0, d["n_err_y"] - TARGET * d["n_auto_y"]).sum()) / d["n_x"].sum()
        return ex, 100 * (d["automation_x"].mean() - d["automation_y"].mean())
    bs = []
    for _ in range(B):
        parts = []
        for per, g in j.groupby("period"):
            g = g.sort_values("month").reset_index(drop=True)
            st = rng.integers(0, len(g) - block + 1, int(np.ceil(len(g) / block)))
            parts.append(g.iloc[np.concatenate([np.arange(s, s + block) for s in st])[:len(g)]])
        bs.append(stat(pd.concat(parts)))
    bs = np.array(bs)
    return stat(j), [np.quantile(bs[:, i], q) for i in (0, 1) for q in (0.025, 0.975)]


cod = R[(R["policy"] == "rolling_cod_known") & (R["alpha"] == 0.10)]
rows = []
for pol, lab in [("rolling_plugin_known", "Rolling plug-in"), ("crc", "Rolling CRC")]:
    a = sel[pol]
    g = R[(R["policy"] == pol) & (R["alpha"] == a)]
    cells = []
    for per in ["A", "B"]:
        ex, au = score(g[g["period"] == per]); cells += [f"{ex:.2f}", f"{au:.1f}"]
    (dex, dau), ci = boot(cod[["period", "month", "n", "n_err", "n_auto", "automation"]], g[["period", "month", "n", "n_err", "n_auto", "automation"]])
    out[f"cod_minus_{pol}"] = {"level": a, "excess_diff": dex, "excess_ci": ci[:2], "auto_diff": dau, "auto_ci": ci[2:]}
    rows.append(f"{lab}, $\\alpha'={a:.2f}$ & " + " & ".join(cells) + f" & ${dex:+.2f}$ [${ci[0]:+.2f}$, ${ci[1]:+.2f}$] & ${dau:+.1f}$ [${ci[2]:+.1f}$, ${ci[3]:+.1f}$] \\\\")
    tag = "pl" if pol == "rolling_plugin_known" else "crc"
    num[f"ds{tag}level"] = f"{a:.2f}"; num[f"ds{tag}exdiff"] = f"{dex:.2f}"; num[f"ds{tag}audiff"] = f"{dau:.1f}"
    num[f"ds{tag}aulo"] = f"{ci[2]:.1f}"; num[f"ds{tag}auhi"] = f"{ci[3]:.1f}"
    num[f"ds{tag}exlo"] = f"{ci[0]:.2f}"; num[f"ds{tag}exhi"] = f"{ci[1]:.2f}"
cells = []
for per in ["A", "B"]:
    ex, au = score(cod[cod["period"] == per]); cells += [f"{ex:.2f}", f"{au:.1f}"]
rows.insert(0, "Rolling CoD, $\\alpha=0.10$ & " + " & ".join(cells) + " & -- & -- \\\\")
(T / "dselected.tex").write_text(r"""\begin{table*}[t]
\caption{Comparison at a margin chosen on the development period. For plug-in and rolling CRC, the nominal level $\alpha'\in\{0.05,0.10,0.15\}$ is selected on period D with the same rule used for the knobs of \cod (maximize automation subject to at most one excess error per 1{,}000 against the 10\%% target, overall and near each placebo); all rows are scored against the 10\%% target. Last two columns: rolling \cod minus the rule, pooled over A and B, with 95\%% three-month block-bootstrap intervals.}
\label{tab:dselected}
\small
\begin{tabular}{l rr rr rr}
\toprule
& \multicolumn{2}{c}{Window A} & \multicolumn{2}{c}{Window B} & \multicolumn{2}{c}{\cod minus rule} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Rule & Excess & Auto. & Excess & Auto. & $\Delta$ excess / 1k & $\Delta$ automation \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % "\n".join(rows))

# 2. within-window pre/post
for pol, tag in [("static_cod", "scod"), ("rolling_cod_known", "rcod"), ("static_plugin", "spl")]:
    for per in ["A", "B"]:
        g = R[(R["policy"] == pol) & (R["alpha"] == 0.10) & (R["period"] == per)]
        for side, gg in [("pre", g[g["month"] < BRK[per]]), ("post", g[g["month"] > BRK[per]])]:
            nv = gg[gg["n_auto"] >= 50]
            ex = 1000 * gg["excess"].sum() / gg["n"].sum()
            num[f"{tag}{per}{side}ex"] = f"{ex:.2f}"; num[f"{tag}{per}{side}sig"] = str(int((nv["ci_lo"] > 0.10).sum()))
            num[f"{tag}{per}{side}n"] = str(len(gg))
            out[f"{pol}|{per}|{side}"] = {"excess": ex, "sig": int((nv["ci_lo"] > 0.10).sum()), "months": len(gg)}

# 3. class-prior shift + tier accuracy
S = pd.read_parquet(data / "samples_v1.parquet", columns=["period", "set", "month", "regime", "product"])
srow, acc = [], json.loads((V / "analysis.json").read_text())["deploy_accuracy"]
for per, (old, new) in {"A": ("R0", "R1"), "B": ("R1", "R2")}.items():
    e = S[(S["period"] == per) & (S["set"] == "eval")]
    pre = e[e["month"].isin([ms(BRK[per], -k) for k in (1, 2, 3)])]
    post = e[e["month"].isin([ms(BRK[per], k) for k in (1, 2, 3)])]
    lab_pre = pre["product"].map(lambda y: map_label(y, old, new) or f"[no carry-over] {y}")
    p = lab_pre.value_counts(normalize=True); q = post["product"].value_counts(normalize=True)
    tv = 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p.index) | set(q.index))
    num[f"tv{per}"] = f"{tv:.3f}"; out[f"tv_{per}"] = tv
    srow.append(f"{per} & {tv:.3f} & {100*acc[f't0|{per}|pre']:.1f} / {100*acc[f't0|{per}|post']:.1f} & "
                f"{100*acc[f'f|{per}|pre']:.1f} / {100*acc[f'f|{per}|post']:.1f} & {100*acc[f't1|{per}|pre']:.1f} / {100*acc[f't1|{per}|post']:.1f} \\\\")
(T / "app_shift.tex").write_text(r"""\begin{table}[t]
\caption{Size of each break. TV: total-variation distance between the product mix of the stream in the three months before and after the break (earlier labels mapped through the crosswalk; labels without carry-over kept as separate categories). Remaining columns: accuracy of each tier on stream complaints before / after the break (\%%).}
\label{tab:shift}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrr}
\toprule
Window & TV & TF-IDF & MiniLM & ModernBERT \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(srow))
WORDS = dict(zip("0123456789", ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]))
mname = lambda k: "".join(WORDS.get(ch, ch) for ch in k if ch.isalnum())
names = [mname(k) for k in num]; assert len(names) == len(set(names))
(paper / "numbers_secondary.tex").write_text("% generated by scripts/26_secondary.py; do not edit\n" + "\n".join(
    r"\newcommand{\M" + mname(k) + "}{" + v + "}" for k, v in num.items()) + "\n")
(V / "secondary.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps(out, indent=1, default=float))
