"""Generate every LaTeX table of the paper from result files (never typed by hand).
Usage: python -I scripts/21_make_tables.py <data_dir> <results_dir> <paper_dir>
Writes paper/tables/*.tex and paper/numbers.tex (\newcommand macros for numbers quoted in text).
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

data, res, paper = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
V = res / "v1"; T = paper / "tables"; T.mkdir(parents=True, exist_ok=True)
S = pd.read_csv(V / "main_summary.csv")
DS = pd.read_csv(V / "dev_factorial_summary.csv")
A = json.loads((V / "analysis.json").read_text())
num = {}

NAME = {"rolling_cod_known": "Rolling CoD, reset at break", "rolling_cod_none": "Rolling CoD, no reset",
        "rolling_cod_det": "Rolling CoD, detector reset", "rolling_plugin_known": "Rolling plug-in, reset",
        "rolling_plugin_none": "Rolling plug-in, no reset", "rolling_plugin_det": "Rolling plug-in, detector reset",
        "static_cod": "Static CoD", "static_plugin": "Static plug-in", "fullstatic_cod": "Fully static CoD",
        "fullstatic_plugin": "Fully static plug-in", "B1_fatal": "B1: static encoder threshold",
        "encoder_only_cod": "Encoder-only rolling CoD", "crc": "Rolling CRC (joint risk)",
        "wcrc": "Weighted CRC (joint risk)", "aci": "ACI"}


def fmt(x, d=2):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


# ---- Table: main Product policies, alpha = 0.10 (A, B) + dev D
order = ["B1_fatal", "static_plugin", "static_cod", "rolling_plugin_known", "crc", "wcrc", "aci",
         "rolling_cod_none", "rolling_cod_det", "encoder_only_cod", "rolling_cod_known"]
p = S[(S["task"] == "product") & (S["split"] == "main") & (S["alpha"] == 0.10)]
d = DS[DS["alpha"] == 0.10]
rows = []
for pol in order:
    cells = []
    for per in ["A", "B"]:
        r = p[(p["period"] == per) & (p["policy"] == pol)]
        r = r.iloc[0] if len(r) else None
        cells += [fmt(r["excess_per_1k"]) if r is not None else "--", str(int(r["ci_lo_over"])) if r is not None else "--",
                  str(int(r["vacuous_months"])) if r is not None else "--", fmt(100 * r["automation"], 1) if r is not None else "--"]
    rd = d[d["policy"] == pol]
    cells += [fmt(rd.iloc[0]["excess_per_1k"]) if len(rd) else "--", fmt(100 * rd.iloc[0]["automation"], 1) if len(rd) else "--"]
    name = NAME[pol]
    if pol == "rolling_cod_known":
        name = r"\textbf{" + name + "}"; cells = [r"\textbf{" + c + "}" for c in cells]
    rows.append(name + " & " + " & ".join(cells) + r" \\")
nominal = p[p["policy"] == "rolling_cod_known"][["period", "ci_lo_expected_nominal"]].set_index("period")["ci_lo_expected_nominal"]
(T / "main_policies.tex").write_text(r"""\begin{table*}[t]
\caption{Product routing at $\alpha=0.10$, cascade TF-IDF $\to$ ModernBERT $\to$ human. Excess: excess errors per 1{,}000 complaints, $1000\sum_m\max(0,e_m-\alpha a_m)/\sum_m n_m$; Sig.: months whose two-sided 95\%% Clopper--Pearson lower bound on the monthly error among auto-routed complaints exceeds $\alpha$ (about 0.025 per non-vacuous month under nominal validity; %s in A and %s in B for the reference rule); Vac.: months with fewer than 50 auto-routed complaints; Auto.: share of complaints routed without a human (\%%). Window A spans the 2017 break (33 months), B the 2023 break (31 months); D is the break-free development period (31 months) on which the knobs were selected; its cascade is TF-IDF $\to$ MiniLM $\to$ human, since ModernBERT is not trained for D.}
\label{tab:main}
\small
\begin{tabular}{l rrrr rrrr rr}
\toprule
& \multicolumn{4}{c}{Window A (2017 break)} & \multicolumn{4}{c}{Window B (2023 break)} & \multicolumn{2}{c}{Dev D (no break)} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-11}
Policy & Excess & Sig. & Vac. & Auto. & Excess & Sig. & Vac. & Auto. & Excess & Auto. \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % (fmt(nominal.get("A"), 2), fmt(nominal.get("B"), 2), "\n".join(rows)))

# ---- Table: contrasts
crow = []
for a in [0.10, 0.05, 0.15]:
    C = pd.read_csv(V / f"contrasts_alpha{a:.2f}.csv")
    for _, r in C.iterrows():
        ex = eval(r["excess_ci95"]) if isinstance(r["excess_ci95"], str) else r["excess_ci95"]
        au = eval(r["automation_ci95"]) if isinstance(r["automation_ci95"], str) else r["automation_ci95"]
        ph = "$<$0.001" if r["p_holm"] < 0.001 else f"{r['p_holm']:.3f}"
        m = lambda v, d: f"${v:+.{d}f}$"
        crow.append(f"{a:.2f} & {r['contrast']} & {m(r['excess_per_1k_diff'],2)} [{m(ex[0],2)}, {m(ex[1],2)}] & "
                    f"{m(100*r['automation_diff'],1)} [{m(100*au[0],2)}, {m(100*au[1],2)}] & {ph} \\\\")
        if a == 0.10:
            num[f"c{r['contrast'][1]}ex"] = f"{r['excess_per_1k_diff']:.1f}"
            num[f"c{r['contrast'][1]}exlo"] = f"{ex[0]:.1f}"; num[f"c{r['contrast'][1]}exhi"] = f"{ex[1]:.1f}"
            num[f"c{r['contrast'][1]}au"] = f"{100*r['automation_diff']:.1f}"
            num[f"c{r['contrast'][1]}aulo"] = f"{100*au[0]:.2f}"; num[f"c{r['contrast'][1]}auhi"] = f"{100*au[1]:.2f}"
            num[f"c{r['contrast'][1]}p"] = ph
            num[f"c{r['contrast'][1]}exabs"] = f"{abs(r['excess_per_1k_diff']):.1f}"
            num[f"c{r['contrast'][1]}auabs"] = f"{abs(100*r['automation_diff']):.1f}"
    crow.append(r"\midrule" if a != 0.15 else "")
(T / "contrasts.tex").write_text(r"""\begin{table}[t]
\caption{Pre-specified contrasts (Product, windows A and B pooled). $\Delta$ is the first policy minus the second; 95\%% intervals from a 3-month block bootstrap (2{,}000 replicates); $p$ is Holm-adjusted over C1--C4 within each $\alpha$ (headline: excess for C1--C3, automation for C4). C1: rolling CoD vs static plug-in; C2: rolling CoD vs rolling plug-in; C3: reset vs no reset, break..break+6; C4: cascade with vs without the LLM tier on the first 1{,}000 stream complaints of each month, break..break+6.}
\label{tab:contrasts}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llrrr}
\toprule
$\alpha$ & & $\Delta$ excess / 1k & $\Delta$ automation (pts) & $p$ \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(x for x in crow if x != ""))

# ---- Table: new-class handling
R = pd.read_csv(V / "main_monthly.csv")
BRK = {"A": "2017-04", "B": "2023-08"}; ms = lambda m, k: str(pd.Period(m, "M") + k)
nrows = []
for pol in ["B1_fatal", "static_plugin", "rolling_cod_none", "rolling_cod_known"]:
    cells = []
    for per in ["A", "B"]:
        g = R[(R["task"] == "product") & (R["split"] == "main") & (R["alpha"] == 0.10) & (R["policy"] == pol) & (R["period"] == per)
              & R["month"].isin([ms(BRK[per], k) for k in range(7)])]
        n = g["n_new"].sum(); ok, bad = g["new_correct_auto"].sum() / n, g["new_wrong_auto"].sum() / n
        cells += [fmt(100 * ok, 1), fmt(100 * bad, 1), fmt(100 * (1 - ok - bad), 1)]
        num[f"new{per}n"] = str(int(n))
        if pol == "rolling_cod_known":
            num[f"new{per}bad"] = fmt(100 * bad, 0); num[f"new{per}def"] = fmt(100 * (1 - ok - bad), 0)
    nrows.append(NAME[pol] + " & " + " & ".join(cells) + r" \\")
(T / "newclass.tex").write_text(r"""\begin{table}[t]
\caption{Complaints whose true product is a class created at the break (A: \emph{Vehicle loan or lease}, $n=%s$; B: \emph{Prepaid card} and \emph{Debt or credit management}, $n=%s$), months break..break+6, $\alpha=0.10$: correctly auto-routed / wrongly auto-routed / sent to a human (\%%).}
\label{tab:newclass}
\small
\setlength{\tabcolsep}{1.8pt}
\begin{tabular}{l rrr rrr}
\toprule
& \multicolumn{3}{c}{Window A} & \multicolumn{3}{c}{Window B} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Policy & OK & Wrong & Human & OK & Wrong & Human \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (num["newAn"], num["newBn"], "\n".join(nrows)))

# ---- Table: cost
CT = pd.read_csv(V / "cost_table.csv")
crows = []
for pol in ["B1_fatal", "static_plugin", "static_cod", "rolling_plugin_known", "rolling_cod_known"]:
    cells = []
    for per in ["A", "B"]:
        for ch in [0.5, 1.5, 3.0]:
            v = CT[(CT["task"] == "product") & (CT["alpha"] == 0.10) & (CT["policy"] == pol) & (CT["period"] == per) & (CT["c_h"] == ch)]["usd_per_1k"]
            cells.append(f"{v.iloc[0]:.0f}")
    crows.append(NAME[pol] + " & " + " & ".join(cells) + r" \\")
aud = CT[(CT["task"] == "product") & (CT["policy"] == "rolling_cod_known") & (CT["alpha"] == 0.10) & (CT["c_h"] == 1.5)].set_index("period")["audit_share_of_traffic"]
num["auditA"] = fmt(100 * aud["A"], 1); num["auditB"] = fmt(100 * aud["B"], 1)
(T / "cost.tex").write_text(r"""\begin{table}[t]
\caption{Human review cost in USD per 1{,}000 complaints at $\alpha=0.10$, with the monthly 600-item audit charged against real monthly narrative volume. Per-item cost $c_h\in\{0.5, 1.5, 3.0\}$ USD corresponds to about 1.4, 4.2 and 8.4 minutes at the May 2025 U.S. median wage of customer service representatives \cite{bls2025oews}. Machine cost on the laptop is negligible in comparison.}
\label{tab:cost}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{l rrr rrr}
\toprule
& \multicolumn{3}{c}{Window A} & \multicolumn{3}{c}{Window B} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Policy & 0.5 & 1.5 & 3.0 & 0.5 & 1.5 & 3.0 \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(crows))

# ---- Appendix: full alpha grid (Product)
full = []
for pol in order:
    cells = []
    for per in ["A", "B"]:
        for a in [0.05, 0.10, 0.15]:
            r = S[(S["task"] == "product") & (S["split"] == "main") & (S["alpha"] == a) & (S["period"] == per) & (S["policy"] == pol)]
            cells.append(f"{r.iloc[0]['excess_per_1k']:.1f}/{100*r.iloc[0]['automation']:.0f}" if len(r) else "--")
    full.append(NAME[pol] + " & " + " & ".join(cells) + r" \\")
(T / "app_alpha_grid.tex").write_text(r"""\begin{table*}[t]
\caption{Product, all $\alpha$: excess errors per 1k / automation (\%%). In window B at $\alpha=0.15$ the pooled machine error is below the target, so accepting everything meets it on average, though not in every month.}
\label{tab:alphagrid}
\small
\begin{tabular}{l rrr rrr}
\toprule
& \multicolumn{3}{c}{Window A} & \multicolumn{3}{c}{Window B} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Policy & 0.05 & 0.10 & 0.15 & 0.05 & 0.10 & 0.15 \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % "\n".join(full))

# ---- Appendix: Issue task
irows = []
for pol in ["static_plugin", "fullstatic_plugin", "static_cod", "rolling_plugin_known", "rolling_cod_known"]:
    cells = []
    for per in ["A", "D", "B"]:
        r = S[(S["task"] == "issue") & (S["alpha"] == 0.10) & (S["period"] == per) & (S["policy"] == pol)]
        cells.append(f"{r.iloc[0]['excess_per_1k']:.1f}/{int(r.iloc[0]['ci_lo_over'])}/{100*r.iloc[0]['automation']:.1f}" if len(r) else "--")
    irows.append(NAME[pol] + " & " + " & ".join(cells) + r" \\")
(T / "app_issue.tex").write_text(r"""\begin{table}[t]
\caption{Issue routing (about 80 classes per regime), $\alpha=0.10$, cascade TF-IDF $\to$ MiniLM $\to$ human: excess per 1k / significant months / automation (\%%).}
\label{tab:issue}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrr}
\toprule
Policy & A & D & B \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(irows))

# ---- Post-hoc matched-margin analysis (added in Phase 5 review; from existing monthly files)
DM = pd.read_csv(V / "dev_factorial_monthly.csv")
RM = R[(R["task"] == "product") & (R["split"] == "main")]
def rescore(g, target=0.10):
    ex = np.maximum(0, g["n_err"] - target * g["n_auto"]).sum() / g["n"].sum() * 1000
    return ex, 100 * g["n_auto"].sum() / g["n"].sum()
mrows, mm_gaps = [], []
for pol, a, lab in [("rolling_cod_known", 0.10, "Rolling CoD, $\\alpha=0.10$"), ("rolling_cod_known", 0.05, "Rolling CoD, $\\alpha=0.05$"),
                    ("rolling_plugin_known", 0.05, "Rolling plug-in, $\\alpha=0.05$"), ("crc", 0.05, "Rolling CRC, $\\alpha=0.05$"),
                    ("static_plugin", 0.05, "Static plug-in, $\\alpha=0.05$"), ("B1_fatal", 0.05, "B1, $\\alpha=0.05$")]:
    cells = []
    for per in ["A", "B"]:
        ex, au = rescore(RM[(RM["policy"] == pol) & (RM["alpha"] == a) & (RM["period"] == per)])
        cells += [f"{ex:.2f}", f"{au:.1f}"]
        if pol == "rolling_plugin_known" and a == 0.05:
            num[f"mmpl{per}ex"] = f"{ex:.2f}"; num[f"mmpl{per}au"] = f"{au:.1f}"
    g = DM[(DM["policy"] == pol) & (DM["alpha"] == a)]
    if len(g):
        ex, au = rescore(g); cells += [f"{ex:.2f}", f"{au:.1f}"]
        if pol == "rolling_plugin_known" and a == 0.05:
            num["mmplDex"] = f"{ex:.2f}"; num["mmplDau"] = f"{au:.1f}"
    else:
        cells += ["--", "--"]
    mrows.append(lab + " & " + " & ".join(cells) + r" \\")
    if a == 0.05 and pol in ("rolling_plugin_known", "crc"):
        for per in ["A", "B", "D"]:
            gg = RM[(RM["policy"] == pol) & (RM["alpha"] == a) & (RM["period"] == per)] if per != "D" else DM[(DM["policy"] == pol) & (DM["alpha"] == a)]
            cc = RM[(RM["policy"] == "rolling_cod_known") & (RM["alpha"] == 0.10) & (RM["period"] == per)] if per != "D" else DM[(DM["policy"] == "rolling_cod_known") & (DM["alpha"] == 0.10)]
            mm_gaps.append(rescore(cc)[1] - rescore(gg)[1])
num["mmgaplo"], num["mmgaphi"] = f"{min(mm_gaps):.0f}", f"{max(mm_gaps):.0f}"
(T / "matched_margin.tex").write_text(r"""\begin{table}[t]
\caption{Post-hoc matched-margin analysis (added after review, computed from the same monthly outputs): policies run at a stricter nominal level and scored against the 10\%% target. Excess errors per 1{,}000 / automation (\%%).}
\label{tab:matched}
\small
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{l rr rr rr}
\toprule
& \multicolumn{2}{c}{A} & \multicolumn{2}{c}{B} & \multicolumn{2}{c}{D} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Run as & Exc. & Auto. & Exc. & Auto. & Exc. & Auto. \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(mrows))

# ---- numbers used in text
mp = S[(S["task"] == "product") & (S["split"] == "main") & (S["alpha"] == 0.10)].set_index(["period", "policy"])
for per in ["A", "B"]:
    for pol, tag in [("rolling_cod_known", "cod"), ("B1_fatal", "bone"), ("static_plugin", "spl"), ("static_cod", "scod"),
                     ("rolling_plugin_known", "rpl"), ("rolling_cod_none", "nores")]:
        r = mp.loc[(per, pol)]
        num[f"{tag}{per}ex"] = fmt(r["excess_per_1k"]); num[f"{tag}{per}sig"] = str(int(r["ci_lo_over"]))
        num[f"{tag}{per}au"] = fmt(100 * r["automation"], 1)
    for pol, tag in [("rolling_cod_known", "cod"), ("rolling_cod_none", "nores")]:
        r = mp.loc[(per, pol)]
        num[f"{tag}{per}expost"] = fmt(r["excess_per_1k_post6"]); num[f"{tag}{per}aupost"] = fmt(100 * r["automation_post6"], 0)
dd = DS[DS["alpha"] == 0.10].set_index("policy")
num["devscodex"] = fmt(dd.loc["static_cod", "excess_per_1k"]); num["devcodau"] = fmt(100 * dd.loc["rolling_cod_known", "automation"], 1)
num["devnoresau"] = fmt(100 * dd.loc["rolling_cod_none", "automation"], 1); num["devcodex"] = fmt(dd.loc["rolling_cod_known", "excess_per_1k"])
st = pd.DataFrame(A["stress_vs_main_alpha0.10"]).set_index(["period", "policy", "split"])
num["stressBex"] = fmt(st.loc[("B", "rolling_cod_known", "stress"), "excess_per_1k"], 1)
num["stressBsig"] = str(int(st.loc[("B", "rolling_cod_known", "stress"), "ci_lo_over"]))
num["stressAex"] = fmt(st.loc[("A", "rolling_cod_known", "stress"), "excess_per_1k"], 2)
c4 = json.loads((V / "c4_drop_rule.json").read_text())["0.1"]
num["cfourmonths"] = str(c4["months_gain_ge_2pts"]); num["cfourof"] = str(c4["of"])
num["cfourgain"] = fmt(c4["mean_gain_pts"], 1)
num["cfournewwith"] = str(c4["new_correct_auto_with"]); num["cfournewwithout"] = str(c4["new_correct_auto_without"])
num["cfournewtot"] = str(c4["new_items"])
ct = A["contamination"]
num["contold"] = fmt(100 * ct["acc_old_era"], 1); num["contnew"] = fmt(100 * ct["acc_new_era_reweighted_to_old_mix"], 1)
iss = S[(S["task"] == "issue") & (S["alpha"] == 0.10) & (S["policy"] == "rolling_cod_known")].set_index("period")
num["issueA"] = fmt(100 * iss.loc["A", "automation"], 1); num["issueB"] = fmt(100 * iss.loc["B", "automation"], 1)
cost = CT[(CT["task"] == "product") & (CT["alpha"] == 0.10) & (CT["c_h"] == 1.5)].set_index(["period", "policy"])["usd_per_1k"]
num["costcodA"] = f"{cost[('A','rolling_cod_known')]:.0f}"; num["costboneA"] = f"{cost[('A','B1_fatal')]:.0f}"
num["costcodB"] = f"{cost[('B','rolling_cod_known')]:.0f}"; num["costboneB"] = f"{cost[('B','B1_fatal')]:.0f}"
nfl = A["noise_floor_alpha0.10"]
for per in ["A", "B"]:
    lo, hi = nfl[f"{per}|rolling_cod_known"]["excess_per_1k_p5_p95"]
    num[f"nf{per}lo"] = f"{lo:.2f}"; num[f"nf{per}hi"] = f"{hi:.2f}"
# issue carry-over by name identity (classes with >= 50 narratives; break months excluded)
import importlib.util
_sp = importlib.util.spec_from_file_location("mt", Path(__file__).parent / "14_monthly_tiers.py"); mt = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(mt)
ISETS = mt.issue_sets(data)
LC = pd.read_parquet(data / "label_counts_narr.parquet"); LC = LC[~LC["month"].isin(["2017-04", "2023-08"])]
LC["regime"] = np.where(LC["month"] < "2017-05", "R0", np.where(LC["month"] < "2023-09", "R1", "R2"))
for a_, b_, tag in [("R0", "R1", "one"), ("R1", "R2", "two")]:
    keep = ISETS[b_] & ISETS[a_]; v = LC[(LC["regime"] == b_) & LC["issue"].isin(ISETS[b_])]
    num[f"iss{tag}carried"] = str(len(keep)); num[f"iss{tag}total"] = str(len(ISETS[b_]))
    num[f"iss{tag}vol"] = f"{100 * v[v['issue'].isin(keep)]['n'].sum() / v['n'].sum():.0f}"
# conformal risk control on its native (joint) target, alpha = 0.10
for pol, tag in [("crc", "crc"), ("wcrc", "wcrc")]:
    for per in ["A", "B"]:
        g = R[(R["task"] == "product") & (R["split"] == "main") & (R["alpha"] == 0.10) & (R["policy"] == pol) & (R["period"] == per)]
        num[f"{tag}joint{per}"] = str(int((g["joint_risk"] > 0.10).sum()))
        num[f"{tag}jointpool{per}"] = f"{100 * g['n_err'].sum() / g['n'].sum():.1f}"
sig = S[(S["task"] == "product") & (S["split"] == "main") & (S["alpha"] == 0.10) & S["policy"].isin(["crc", "wcrc"]) & S["period"].isin(["A", "B"])]["ci_lo_over"]
num["crcsiglo"], num["crcsighi"] = str(int(sig.min())), str(int(sig.max()))
# largest monthly automation gain of the LLM tier (break..break+6, alpha = 0.10)
gains = []
for per in ["A", "B"]:
    w = R[(R["policy"] == "C4_with_llm") & (R["alpha"] == 0.10) & (R["period"] == per)].set_index("month")["automation"]
    o = R[(R["policy"] == "C4_without_llm") & (R["alpha"] == 0.10) & (R["period"] == per)].set_index("month")["automation"]
    for m in [ms(BRK[per], k) for k in range(7)]:
        gains.append((100 * (w[m] - o[m]), per, m, 100 * R[(R["policy"] == "C4_with_llm") & (R["alpha"] == 0.10) & (R["period"] == per) & (R["month"] == m)]["auto_t2"].iloc[0]))
g = max(gains)
num["llmmaxgain"] = f"{g[0]:.1f}"; num["llmmaxgainmonth"] = g[2]
num["llmmaxshare"] = f"{max(x[3] for x in gains):.0f}"
# compute (machine-hours) from outputs
llm_h = sum(pd.read_parquet(V / f"llm_scores_{w}.parquet")["sec"].sum() for w in "AB") / 3600
mb_h = sum(v["train_s"] for w in "AB" for v in json.loads((V / f"mbert_meta_{w}.json").read_text()).values()) / 3600
num["llmhours"] = f"{llm_h:.1f}"; num["mbhours"] = f"{mb_h:.1f}"
num["rplDex"] = fmt(DS[(DS["alpha"] == 0.10) & (DS["policy"] == "rolling_plugin_known")].iloc[0]["excess_per_1k"], 1)
dedup = json.loads((res / "dedup_stats_v1.json").read_text())
num["dupshare"] = fmt(100 * dedup["share_in_dup_groups"], 1)
WORDS = dict(zip("0123456789", ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]))
mname = lambda k: "".join(WORDS.get(ch, ch) for ch in k if ch.isalnum())
macros = "\n".join(r"\newcommand{\N" + mname(k) + "}{" + v + "}" for k, v in num.items())
# guard against macro-name collisions
names = [mname(k) for k in num]
assert len(names) == len(set(names)), [n for n in names if names.count(n) > 1]
(paper / "numbers.tex").write_text("% generated by scripts/21_make_tables.py; do not edit\n" + macros + "\n")
print(f"{len(num)} number macros; tables:", sorted(p.name for p in T.glob('*.tex')))
