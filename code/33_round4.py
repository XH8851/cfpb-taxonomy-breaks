r"""Review round 4 (post-hoc, 2026-11-13): noise floor of the clipped metric, SGR baseline, and the
risk-automation frontier of rolling rules with a comparison at matched automation.
Inputs : results/v1/frontier_monthly.csv (32_frontier_sgr.py), results/v1/main_monthly.csv
Outputs: results/v1/round4.json, paper/numbers_round4.tex (macros \R...), paper/tables/frontier.tex,
         figures/fig_frontier_curves.pdf
Definitions
  excess vs target   1000 * sum_m max(0, e_m - 0.10 a_m) / sum_m n_m (all rows scored against 10%)
  floor              expected excess of a rule whose monthly risk equals the target exactly, given the
                     realized automated counts a_m: sum_m E[max(0, X_m - 0.10 a_m)], X_m ~ Bin(a_m, 0.10)
  matched            comparator's excess interpolated (linear, along its nominal levels sorted by
                     automation) at rolling CoD's automation in each window (ex post: uses the window)
  CIs                paired 3-month block bootstrap over months within each window, 2,000 replicates
Usage: python -I scripts/33_round4.py <project_root>
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1]); V = root / "results" / "v1"; P = root / "paper"; FIG = root / "figures"
T = 0.10; B = 2000; BLOCK = 3
rng = np.random.default_rng(20261113)
F = pd.read_csv(V / "frontier_monthly.csv")
MM = pd.read_csv(V / "main_monthly.csv")
MM = MM[(MM["task"] == "product") & (MM["split"] == "main") & MM["period"].isin(["A", "B"])]
COMP = {"rolling_plugin_known": "Plug-in", "crc": "CRC", "rolling_sgr_known": "SGR-style"}
out, num = {}, {}


def floor_months(a):
    a = np.asarray(a, int); f = np.zeros(len(a))
    for i, x in enumerate(a):
        if x:
            k = np.arange(x + 1); f[i] = np.sum(np.maximum(0, k - T * x) * binom.pmf(k, x, T))
    return f


# ---- month-level arrays per window: E[pol][level] (months), A[...], N (months)
W = {}
for per in ["A", "B"]:
    g = F[F["period"] == per]
    months = sorted(g["month"].unique())
    N = g[(g["policy"] == "rolling_cod_known") & (g["alpha"] == 0.10)].set_index("month").loc[months, "n"].values.astype(float)
    arr = {}
    for (pol, lev), h in g.groupby(["policy", "alpha"]):
        h = h.set_index("month").loc[months]
        assert (h["n"].values == N).all()
        arr[(pol, round(lev, 4))] = (h["n_err"].values.astype(float), h["n_auto"].values.astype(float))
    W[per] = dict(months=months, N=N, arr=arr, levels=sorted({k[1] for k in arr}))


def stats(per, idx):
    """Return dict of per-(pol, level) automation (%) and excess (per 1k vs 10%) on month indices idx."""
    w = W[per]; N = w["N"][idx]; res = {}
    for k, (e, a) in w["arr"].items():
        ee, aa = e[idx], a[idx]
        res[k] = (100 * np.mean(aa / N), 1000 * np.maximum(0, ee - T * aa).sum() / N.sum(), 100 * ee.sum() / max(1.0, aa.sum()))
    return res, N.sum()


def matched(per, st, pol):
    """Comparator interpolated to CoD's automation: (excess, nominal level, CoD excess, CoD automation, inside, risk, CoD risk)."""
    au_c, ex_c, r_c = st[("rolling_cod_known", 0.10)]
    lv = W[per]["levels"]
    au = np.array([st[(pol, l)][0] for l in lv]); ex = np.array([st[(pol, l)][1] for l in lv]); rk = np.array([st[(pol, l)][2] for l in lv]); lvv = np.array(lv)
    o = np.argsort(au, kind="stable")
    inside = au.min() <= au_c <= au.max()
    return np.interp(au_c, au[o], ex[o]), np.interp(au_c, au[o], lvv[o]), ex_c, au_c, inside, np.interp(au_c, au[o], rk[o]), r_c


def blocks(n):
    st = rng.integers(0, n - BLOCK + 1, int(np.ceil(n / BLOCK)))
    return np.concatenate([np.arange(s, s + BLOCK) for s in st])[:n]


# point estimates
pt = {per: stats(per, np.arange(len(W[per]["months"]))) for per in ["A", "B"]}
rows = []
for per in ["A", "B"]:
    st, _ = pt[per]
    for pol in ["rolling_cod_known", "rolling_sgr_known"]:
        out[f"{pol}|{per}|0.10"] = {"auto": st[(pol, 0.10)][0], "excess": st[(pol, 0.10)][1]}
    for pol in COMP:
        e, l, ec, ac, ins, rk, rc = matched(per, st, pol)
        out[f"matched|{pol}|{per}"] = {"excess": e, "nominal": l, "cod_excess": ec, "cod_auto": ac, "inside_range": bool(ins), "risk": rk, "cod_risk": rc}
        assert ins, (pol, per)
# bootstrap: matched differences (comparator - CoD) per window and pooled; SGR - CoD at nominal 0.10
bs = {pol: {"A": [], "B": [], "pool": [], "rA": [], "rB": []} for pol in COMP}
bs_sgr = {"ex": [], "au": []}
nA, nB = len(W["A"]["months"]), len(W["B"]["months"])
for _ in range(B):
    ia, ib = blocks(nA), blocks(nB)
    (sa, Na), (sb, Nb) = stats("A", ia), stats("B", ib)
    for pol in COMP:
        ea, _, eca, _, _, ra, rca = matched("A", sa, pol); eb, _, ecb, _, _, rb, rcb = matched("B", sb, pol)
        bs[pol]["rA"].append(rca - ra); bs[pol]["rB"].append(rcb - rb)
        bs[pol]["A"].append(ea - eca); bs[pol]["B"].append(eb - ecb)
        bs[pol]["pool"].append(((ea - eca) * Na + (eb - ecb) * Nb) / (Na + Nb))
    # SGR - CoD at nominal 0.10, pooled excess and pooled mean monthly automation
    exd = ((sa[("rolling_sgr_known", 0.10)][1] - sa[("rolling_cod_known", 0.10)][1]) * Na + (sb[("rolling_sgr_known", 0.10)][1] - sb[("rolling_cod_known", 0.10)][1]) * Nb) / (Na + Nb)
    aud = ((sa[("rolling_sgr_known", 0.10)][0] - sa[("rolling_cod_known", 0.10)][0]) * nA + (sb[("rolling_sgr_known", 0.10)][0] - sb[("rolling_cod_known", 0.10)][0]) * nB) / (nA + nB)
    bs_sgr["ex"].append(exd); bs_sgr["au"].append(aud)
q = lambda x: (float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975)))
NA, NB = W["A"]["N"].sum(), W["B"]["N"].sum()
for pol in COMP:
    da = out[f"matched|{pol}|A"]["excess"] - out[f"matched|{pol}|A"]["cod_excess"]
    db = out[f"matched|{pol}|B"]["excess"] - out[f"matched|{pol}|B"]["cod_excess"]
    dp = (da * NA + db * NB) / (NA + NB)
    out[f"matched_diff|{pol}"] = {"A": da, "B": db, "pooled": dp, "ci_A": q(bs[pol]["A"]), "ci_B": q(bs[pol]["B"]), "ci_pooled": q(bs[pol]["pool"])}
    for per in ["A", "B"]:
        m = out[f"matched|{pol}|{per}"]
        out[f"matched_riskgap|{pol}|{per}"] = {"cod_minus_comp_points": m["cod_risk"] - m["risk"], "ci": q(bs[pol]["r" + per])}
sA, sB = pt["A"][0], pt["B"][0]
sgr_ex = ((sA[("rolling_sgr_known", 0.10)][1] - sA[("rolling_cod_known", 0.10)][1]) * NA + (sB[("rolling_sgr_known", 0.10)][1] - sB[("rolling_cod_known", 0.10)][1]) * NB) / (NA + NB)
sgr_au = ((sA[("rolling_sgr_known", 0.10)][0] - sA[("rolling_cod_known", 0.10)][0]) * nA + (sB[("rolling_sgr_known", 0.10)][0] - sB[("rolling_cod_known", 0.10)][0]) * nB) / (nA + nB)
out["sgr_minus_cod_0.10"] = {"excess": sgr_ex, "excess_ci": q(bs_sgr["ex"]), "auto": sgr_au, "auto_ci": q(bs_sgr["au"])}

# ---- noise floor for the policies of the main table at alpha = 0.10, and realized pooled risk
fl = {}
for (pol, per), g in MM[MM["alpha"] == 0.10].groupby(["policy", "period"]):
    fl[(pol, per)] = 1000 * floor_months(g["n_auto"]).sum() / g["n"].sum()
    out[f"floor|{pol}|{per}"] = fl[(pol, per)]
    out[f"risk|{pol}|{per}"] = float(g["n_err"].sum() / max(1, g["n_auto"].sum()))
main_pols = ["B1_fatal", "static_plugin", "static_cod", "rolling_plugin_known", "crc", "wcrc", "aci", "rolling_cod_none", "rolling_cod_det", "encoder_only_cod", "rolling_cod_known"]  # rows of Table main
vals = [fl[(p, per)] for p in main_pols for per in ["A", "B"]]
num["floorlo"], num["floorhi"] = f"{min(vals):.1f}", f"{max(vals):.1f}"
for per in ["A", "B"]:
    num[f"floorcod{per}"] = f"{fl[('rolling_cod_known', per)]:.1f}"
    num[f"riskcod{per}"] = f"{100 * out[f'risk|rolling_cod_known|{per}']:.1f}"
    num[f"riskspl{per}"] = f"{100 * out[f'risk|static_plugin|{per}']:.1f}"
    num[f"riskrpl{per}"] = f"{100 * out[f'risk|rolling_plugin_known|{per}']:.1f}"
    num[f"floorrpl{per}"] = f"{fl[('rolling_plugin_known', per)]:.1f}"
    num[f"riskcrc{per}"] = f"{100 * out[f'risk|crc|{per}']:.1f}"
    num[f"risksgr{per}"] = f"{pt[per][0][('rolling_sgr_known', 0.10)][2]:.1f}"
    num[f"aboverpl{per}"] = f"{MM[(MM['alpha'] == 0.10) & (MM['policy'] == 'rolling_plugin_known') & (MM['period'] == per)].pipe(lambda g: 1000 * np.maximum(0, g['n_err'] - T * g['n_auto']).sum() / g['n'].sum()) - fl[('rolling_plugin_known', per)]:.1f}"
    num[f"sgrau{per}"] = f"{out[f'rolling_sgr_known|{per}|0.10']['auto']:.1f}"
    num[f"sgrex{per}"] = f"{out[f'rolling_sgr_known|{per}|0.10']['excess']:.2f}"
spl_ex_A = MM[(MM["alpha"] == 0.10) & (MM["policy"] == "static_plugin") & (MM["period"] == "A")].pipe(lambda g: 1000 * np.maximum(0, g["n_err"] - T * g["n_auto"]).sum() / g["n"].sum())
num["splfloormult"] = f"{spl_ex_A / fl[('static_plugin', 'A')]:.0f}"
num["sgrcodex"] = f"{sgr_ex:+.2f}"; num["sgrcodexlo"], num["sgrcodexhi"] = (f"{x:+.2f}" for x in out["sgr_minus_cod_0.10"]["excess_ci"])
num["sgrcodau"] = f"{sgr_au:+.1f}"; num["sgrcodaulo"], num["sgrcodauhi"] = (f"{x:+.1f}" for x in out["sgr_minus_cod_0.10"]["auto_ci"])
tag = {"rolling_plugin_known": "pl", "crc": "crc", "rolling_sgr_known": "sgr"}
for pol, t in tag.items():
    d = out[f"matched_diff|{pol}"]
    num[f"md{t}"] = f"{d['pooled']:+.2f}"; num[f"md{t}lo"], num[f"md{t}hi"] = (f"{x:+.2f}" for x in d["ci_pooled"])
    for per in ["A", "B"]:
        num[f"mlev{t}{per}"] = f"{out[f'matched|{pol}|{per}']['nominal']:.3f}"
# curves compared at fixed automation beyond the 0.10 operating point (A 70%, B 90%)
for per, x in [("A", 70.0), ("B", 90.0)]:
    st = pt[per][0]; lv = W[per]["levels"]
    for pol, t in [("rolling_cod_known", "cod"), ("rolling_sgr_known", "sgr"), ("rolling_plugin_known", "pl"), ("crc", "crc")]:
        au = np.array([st[(pol, l)][0] for l in lv]); ex = np.array([st[(pol, l)][1] for l in lv]); o = np.argsort(au, kind="stable")
        assert au.min() <= x <= au.max(), (pol, per)
        v = float(np.interp(x, au[o], ex[o])); out[f"fixed_auto|{pol}|{per}|{x}"] = v; num[f"fx{t}{per}"] = f"{v:.1f}"
gaps = []
for pol, t in tag.items():
    for per in ["A", "B"]:
        gq = out[f"matched_riskgap|{pol}|{per}"]
        num[f"mrd{t}{per}"] = f"{gq['cod_minus_comp_points']:.1f}"; num[f"mrd{t}{per}lo"], num[f"mrd{t}{per}hi"] = (f"{x:.1f}" for x in gq["ci"])
        num[f"mrisk{t}{per}"] = f"{out[f'matched|{pol}|{per}']['risk']:.1f}"
        if pol != "rolling_sgr_known":
            gaps.append(gq["cod_minus_comp_points"])
num["mrdlo"], num["mrdhi"] = f"{min(gaps):.1f}", f"{max(gaps):.1f}"
# full deferral of CoD after resets vs rolling plug-in at the grid level nearest its matched nominal level
dm, dp = [], []
for per in ["A", "B"]:
    g = F[F["period"] == per]
    c = g[(g["policy"] == "rolling_cod_known") & (g["alpha"] == 0.10)]
    lv = min(W[per]["levels"], key=lambda l: abs(l - out[f"matched|rolling_plugin_known|{per}"]["nominal"]))
    pl = g[(g["policy"] == "rolling_plugin_known") & np.isclose(g["alpha"], lv)].set_index("month")
    first = W[per]["months"][0]
    for m in c[(c["automation"] == 0) & (c["month"] != first)]["month"]:   # first eval month: no released labels for any rule
        dm.append(m); dp.append(100 * pl.loc[m, "automation"])
    out[f"auto_sd|{per}"] = {"cod": float(100 * c["automation"].std()), "plugin_matched": float(100 * pl["automation"].std())}
out["deferral"] = {"months": dm, "plugin_matched_automation": dp}
dA = [(m, x) for m, x in zip(dm, dp) if m < "2019"]   # window A only: in B the deferred month carries a high plug-in risk
num["defmonthsA"] = " and ".join(m for m, _ in dA); num["defplautoA"] = " and ".join(f"{x:.0f}" for _, x in dA)
num["sdcodA"] = f"{out['auto_sd|A']['cod']:.0f}"; num["sdplA"] = f"{out['auto_sd|A']['plugin_matched']:.0f}"
num["mdmaxabs"] = f"{max(abs(out[f'matched_diff|{p}']['pooled']) for p in COMP):.2f}"
lvls = [float(num[f"mlev{t}{per}"]) for t in ["pl", "crc"] for per in ["A", "B"]]
num["mlevlo"], num["mlevhi"] = f"{min(lvls):.3f}", f"{max(lvls):.3f}"

# ---- table
trows = []
c = {per: out[f"rolling_cod_known|{per}|0.10"] for per in ["A", "B"]}
trows.append(r"\cod, $\alpha=0.10$ & 0.10 & %.1f & %.2f & 0.10 & %.1f & %.2f \\" % (pt["A"][0][("rolling_cod_known", 0.10)][2], c["A"]["excess"], pt["B"][0][("rolling_cod_known", 0.10)][2], c["B"]["excess"]))
s = {per: out[f"rolling_sgr_known|{per}|0.10"] for per in ["A", "B"]}
trows.append(r"SGR-style, $\alpha=0.10$ & 0.10 & %.1f & %.2f & 0.10 & %.1f & %.2f \\" % (pt["A"][0][("rolling_sgr_known", 0.10)][2], s["A"]["excess"], pt["B"][0][("rolling_sgr_known", 0.10)][2], s["B"]["excess"]))
trows.append(r"\midrule")
for pol, lab in COMP.items():
    m = {per: out[f"matched|{pol}|{per}"] for per in ["A", "B"]}
    d = out[f"matched_diff|{pol}"]
    trows.append(r"%s, matched & %.3f & %.1f & %.2f & %.3f & %.1f & %.2f \\" % (
        lab, m["A"]["nominal"], m["A"]["risk"], m["A"]["excess"], m["B"]["nominal"], m["B"]["risk"], m["B"]["excess"]))
(P / "tables" / "frontier.tex").write_text(r"""\begin{table}[t]
\caption{Rolling rules at matched automation (product, scored against the 10\%% target; post hoc, Appendix~\ref{app:history}). Risk: pooled selective risk (\%%); Exc.: excess errors per 1{,}000. Upper rows: \cod (automation %s\%% in A, %s\%% in B) and the SGR-style baseline (%s\%%, %s\%%) at nominal $\alpha=0.10$. Lower rows: rolling plug-in, rolling CRC and SGR interpolated along their nominal levels (0.030--0.120) to the automation of \cod in each window; $\alpha'$ is the nominal level that reaches it, found ex post. A rule whose monthly risk equals the target scores %s (A) and %s (B) excess errors at this automation (noise floor of the clipped metric), so the excess column cannot rank these rows.}
\label{tab:frontier}
\small
\setlength{\tabcolsep}{2pt}
\begin{tabular}{l rrr rrr}
\toprule
& \multicolumn{3}{c}{Window A} & \multicolumn{3}{c}{Window B} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Rule & $\alpha'$ & Risk & Exc. & $\alpha'$ & Risk & Exc. \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (f"{c['A']['auto']:.1f}", f"{c['B']['auto']:.1f}", f"{s['A']['auto']:.1f}", f"{s['B']['auto']:.1f}", num["floorcodA"], num["floorcodB"], "\n".join(trows)))

# ---- figure: frontier curves per window
STY = {"rolling_cod_known": ("CoD", "#1f4e9c", "o"), "rolling_sgr_known": ("SGR", "#2a9d8f", "s"),
       "rolling_plugin_known": ("rolling plug-in", "#e07a1f", "^"), "crc": ("rolling CRC", "#8e44ad", "v")}
fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.95), sharey=True)
for ax, per in zip(axes, ["A", "B"]):
    st, _ = pt[per]; lv = W[per]["levels"]
    for pol, (lab, col, mk) in STY.items():
        au = [st[(pol, l)][0] for l in lv]; ex = [st[(pol, l)][1] for l in lv]
        ax.plot(au, ex, "-", color=col, lw=0.9, label=lab)
        ax.plot(st[(pol, 0.10)][0], st[(pol, 0.10)][1], mk, color=col, ms=3.5, mec="k", mew=0.4)
    # floor of a rule exactly at target, along CoD's levels
    w = W[per]; fa, ff = [], []
    for l in lv:
        e, a = w["arr"][("rolling_cod_known", l)]
        fa.append(100 * np.mean(a / w["N"])); ff.append(1000 * floor_months(a).sum() / w["N"].sum())
    ax.plot(fa, ff, "--", color="0.45", lw=0.8, label="at target (floor)")
    for pol, mk, lab in [("static_plugin", "x", "static plug-in"), ("B1_fatal", "+", "B1")]:
        g = MM[(MM["alpha"] == 0.10) & (MM["policy"] == pol) & (MM["period"] == per)]
        ax.plot(100 * g["automation"].mean(), 1000 * np.maximum(0, g["n_err"] - T * g["n_auto"]).sum() / g["n"].sum(), mk, color="k", ms=4, label=lab)
    ax.set_yscale("symlog", linthresh=1); ax.set_xlim(25, 101); ax.set_title(f"window {per}", fontsize=7)
    ax.set_xlabel("automation (%)", fontsize=7); ax.tick_params(labelsize=6)
axes[0].set_ylabel("excess errors per 1k", fontsize=7)
axes[1].legend(fontsize=4.6, frameon=False, loc="upper left", ncol=2, columnspacing=0.6, handlelength=1.4)
fig.tight_layout(pad=0.3, w_pad=0.4); fig.savefig(FIG / "fig_frontier_curves.pdf"); fig.savefig(FIG / "fig_frontier_curves.png", dpi=200)

WORDS = dict(zip("0123456789", ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]))
mname = lambda k: "".join(WORDS.get(ch, ch) for ch in k if ch.isalnum())
names = [mname(k) for k in num]; assert len(names) == len(set(names))
(P / "numbers_round4.tex").write_text("% generated by scripts/33_round4.py; do not edit\n" + "\n".join(
    r"\newcommand{\R" + mname(k) + "}{" + v + "}" for k, v in num.items()) + "\n")
(V / "round4.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps(num, indent=0))
