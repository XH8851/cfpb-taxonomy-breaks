"""Phase 3 analyses for RESULTS_FULL.md (all from cached outputs; CPU only).

  1. contamination probe: LLM zero-shot accuracy, R2 items 2023-08..2024-07 vs 2025-09..2026-08,
     raw and reweighted to the older era's class mix
  2. cost per 1k complaints (volume-scaled): human cost = ((1-automation)*V_m + 600)*c_h / V_m
  3. months to recovery after each break (3-month smoothed selective risk <= alpha thereafter)
  4. noise floor: 20 audit-resampling replays (SD of monthly risk; spread of excess per 1k)
  5. Issue task summary; template-stress split vs main
  6. tier accuracy on deploy items
  7. figures: monthly selective risk (A, B) and the automation-vs-excess frontier
Usage: python -I scripts/20_analysis.py <data_dir> <results_dir>
Outputs: results/v1/analysis.json, results/v1/cost_table.csv, figures/fig_*.pdf|png
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data, res = Path(sys.argv[1]), Path(sys.argv[2])
V = res / "v1"; FIG = res.parent / "figures"; FIG.mkdir(exist_ok=True)
R = pd.read_csv(V / "main_monthly.csv")
S = pd.read_csv(V / "main_summary.csv")
BRK = {"A": "2017-04", "B": "2023-08"}
out = {}
ms = lambda m, k: str(pd.Period(m, "M") + k)

# 1. contamination
C = pd.read_parquet(V / "contam_scores.parquet").merge(pd.read_parquet(V / "contam_items.parquet")[["cid", "era"]], on="cid")
old, new = C[C["era"].str.startswith("2023")], C[C["era"].str.startswith("2025")]
w = old["product"].value_counts(normalize=True)
pc_new = new.groupby("product")["correct"].mean()
re = float(sum(w.get(k, 0) * pc_new.get(k, np.nan) for k in w.index if k in pc_new.index) / sum(w.get(k, 0) for k in w.index if k in pc_new.index))
out["contamination"] = {"acc_old_era": round(old["correct"].mean(), 4), "acc_new_era_raw": round(new["correct"].mean(), 4),
                        "acc_new_era_reweighted_to_old_mix": round(re, 4),
                        "gap_pts_reweighted": round(100 * (old["correct"].mean() - re), 2),
                        "flag_gt_5pts": bool(100 * (old["correct"].mean() - re) > 5)}

# 2. cost (volume-scaled)
L = pd.read_parquet(data / "label_counts_narr.parquet").groupby("month")["n"].sum()
rows = []
for (task, split, per, pol, a), g in R[R["split"] == "main"].groupby(["task", "split", "period", "policy", "alpha"]):
    g = g.copy(); g["V"] = g["month"].map(L)
    for ch in (0.5, 1.5, 3.0):
        cost = (((1 - g["automation"]) * g["V"] + 600) * ch / g["V"] * 1000).mean()
        rows.append({"task": task, "period": per, "policy": pol, "alpha": a, "c_h": ch, "usd_per_1k": cost,
                     "audit_share_of_traffic": (600 / g["V"]).mean()})
CT = pd.DataFrame(rows); CT.to_csv(V / "cost_table.csv", index=False)
key = CT[(CT["task"] == "product") & (CT["alpha"] == 0.10) & (CT["c_h"] == 1.5) &
         CT["policy"].isin(["rolling_cod_known", "static_plugin", "B1_fatal", "rolling_plugin_known", "static_cod"])]
out["cost_usd_per_1k_ch1.5_alpha0.10"] = key.pivot_table(index="policy", columns="period", values="usd_per_1k").to_dict()

# 3. recovery months
rec = {}
for (per, pol, a), g in R[(R["task"] == "product") & (R["split"] == "main")].groupby(["period", "policy", "alpha"]):
    g = g.sort_values("month").set_index("month")
    post = [m for m in g.index if m >= BRK[per]]
    sm = g["risk"].fillna(0).rolling(3, min_periods=1).mean()
    r = None
    for i, m in enumerate(post):
        if (sm.loc[post[i:]] <= a).all():
            r = i; break
    rec[f"{per}|{pol}|{a}"] = r
out["months_to_recovery"] = {k: v for k, v in rec.items() if k.split("|")[1] in
                             ("rolling_cod_known", "rolling_cod_none", "static_cod", "static_plugin", "B1_fatal", "rolling_plugin_known")}

# 4. noise floor
RS = pd.read_csv(V / "resample_monthly.csv")
nf = {}
for (per, pol), g in RS.groupby(["period", "policy"]):
    sd = g.groupby("month")["risk"].std().median()
    ex = g.groupby("seed").apply(lambda x: 1000 * x["excess"].sum() / x["n"].sum())
    au = g.groupby("seed")["automation"].mean()
    nf[f"{per}|{pol}"] = {"median_monthly_risk_sd_pts": round(100 * sd, 2), "excess_per_1k_p5_p95": [round(ex.quantile(.05), 2), round(ex.quantile(.95), 2)],
                          "automation_p5_p95": [round(au.quantile(.05), 3), round(au.quantile(.95), 3)]}
out["noise_floor_alpha0.10"] = nf

# 5. Issue + stress
cols = ["period", "policy", "excess_per_1k", "ci_lo_over", "ci_lo_expected_nominal", "vacuous_months", "automation"]
out["issue_alpha0.10"] = S[(S["task"] == "issue") & (S["alpha"] == 0.10) & S["policy"].isin(
    ["rolling_cod_known", "rolling_cod_none", "rolling_plugin_known", "static_cod", "static_plugin", "fullstatic_plugin"])][cols].to_dict(orient="records")
st = S[(S["task"] == "product") & (S["alpha"] == 0.10) & S["policy"].isin(["rolling_cod_known", "static_plugin"])]
out["stress_vs_main_alpha0.10"] = st[["period", "policy", "split", "excess_per_1k", "ci_lo_over", "automation"]].to_dict(orient="records")

# 6. tier accuracy on deploy items
acc = {}
for name, f in [("t0", "tier_tfidf_product.parquet"), ("f", "tier_emb_product.parquet")]:
    d = pd.read_parquet(V / f, columns=["period", "role", "deploy_month", "correct"]); d = d[d["role"] == "deploy"]
    for per in ["A", "B", "D"]:
        x = d[d["period"] == per]
        if per in BRK:
            acc[f"{name}|{per}|pre"] = round(x[x["deploy_month"] < BRK[per]]["correct"].mean(), 4)
            acc[f"{name}|{per}|post"] = round(x[x["deploy_month"] > BRK[per]]["correct"].mean(), 4)
        else:
            acc[f"{name}|{per}|all"] = round(x["correct"].mean(), 4)
for per in ["A", "B"]:
    d = pd.read_parquet(V / f"tier_mbert_product_{per}.parquet", columns=["role", "deploy_month", "correct"]); d = d[d["role"] == "deploy"]
    acc[f"t1|{per}|pre"] = round(d[d["deploy_month"] < BRK[per]]["correct"].mean(), 4)
    acc[f"t1|{per}|post"] = round(d[d["deploy_month"] > BRK[per]]["correct"].mean(), 4)
    l = pd.read_parquet(V / f"tier_llm_product_{per}.parquet", columns=["role", "deploy_month", "correct"]); l = l[l["role"] == "deploy"]
    acc[f"t2|{per}|scoped"] = round(l["correct"].mean(), 4)
out["deploy_accuracy"] = acc
(V / "analysis.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))

# 7. figures
plt.rcParams.update({"font.size": 8, "figure.dpi": 150})
pols = [("B1_fatal", "B1: static encoder threshold", "#b2182b"), ("static_plugin", "Static plug-in", "#ef8a62"),
        ("static_cod", "Static CoD", "#999999"), ("rolling_plugin_known", "Rolling plug-in, reset", "#67a9cf"),
        ("rolling_cod_known", "Rolling CoD, reset", "#2166ac")]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
for ax, per in zip(axes, ["A", "B"]):
    for pol, lab, col in pols:
        g = R[(R["task"] == "product") & (R["split"] == "main") & (R["alpha"] == 0.10) & (R["period"] == per) & (R["policy"] == pol)].sort_values("month")
        x = pd.PeriodIndex(g["month"], freq="M").to_timestamp()
        ax.plot(x, 100 * g["risk"], lw=1.1, color=col, label=lab)
    ax.axhline(10, color="k", lw=0.6, ls="--")
    ax.axvline(pd.Timestamp(BRK[per] + "-01"), color="k", lw=0.6, ls=":")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelsize=6, rotation=30)
    ax.set_title(f"Window {per} (break {'2017-04-24' if per == 'A' else '2023-08-25'})")
    ax.set_ylabel("monthly error among auto-routed (%)") if per == "A" else None
axes[0].legend(fontsize=6, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "fig_monthly_risk.pdf"); fig.savefig(FIG / "fig_monthly_risk.png")
fig, ax = plt.subplots(figsize=(3.4, 2.7))
sub = S[(S["task"] == "product") & (S["split"] == "main") & (S["alpha"] == 0.10)]
FR = [("B1_fatal", "B1", "#b2182b", "X"), ("static_plugin", "Static plug-in", "#ef8a62", "v"), ("static_cod", "Static CoD", "#999999", "^"),
      ("rolling_plugin_known", "Rolling plug-in", "#67a9cf", "D"), ("crc", "Rolling CRC", "#8c6bb1", "P"), ("aci", "ACI", "#41ae76", "*"),
      ("rolling_cod_none", "Rolling CoD, no reset", "#6baed6", "s"), ("rolling_cod_known", "Rolling CoD, reset", "#2166ac", "o")]
for pol, lab, colr, mk in FR:
    for per, fill in [("A", colr), ("B", "none")]:
        g = sub[(sub["period"] == per) & (sub["policy"] == pol)]
        ax.scatter(100 * g["automation"], g["excess_per_1k"], marker=mk, s=28, facecolors=fill, edgecolors=colr,
                   label=lab if per == "A" else None)
ax.set_yscale("symlog", linthresh=1); ax.set_xlabel("automation (%)"); ax.set_ylabel("excess errors per 1k")
ax.set_title("filled: window A, open: window B", fontsize=7)
ax.legend(fontsize=5.5, frameon=False, loc="upper left", ncol=2)
fig.tight_layout(); fig.savefig(FIG / "fig_frontier.pdf"); fig.savefig(FIG / "fig_frontier.png")
print("figures written")
