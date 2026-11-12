"""SPEC v1 main replays (run ONCE on windows A and B, after knobs are frozen on D).

Product (A, B): factorial + baselines on cascade T0 -> T1 (all eval months, 2,000 items/month);
  fatal baseline B1 = fullstatic_plugin on [t1]; encoder-only rolling_cod_known on [t1];
  C4: [t0,t1,t2] vs [t0,t1] under rolling_cod_known on rank<1000 items, months break-2..break+6.
Issue (A, D, B): factorial + baselines on T0 -> F.
Stress: group-disjoint split (eval items whose near-duplicate group has any training item dropped).
Noise floor: 20 audit-resampling replays for 3 policies at alpha = 0.10.

Usage: python -I scripts/18_replay_v1.py <data_dir> <results_dir> [--no-llm | --only-llm]
  --no-llm   everything except C4 (run while the LLM tier is still being scored)
  --only-llm only the C4 contrast; appended to main_monthly.csv, summary recomputed
Outputs: results/v1/main_monthly.csv, main_summary.csv, resample_monthly.csv
"""
import sys, json, itertools, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

here = Path(__file__).parent
spec = importlib.util.spec_from_file_location("h", here / "16_harness.py"); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
sys.path.insert(0, str(here))
from labels import NEW_CLASSES

data, res = Path(sys.argv[1]), Path(sys.argv[2])
use_llm = "--no-llm" not in sys.argv
only_llm = "--only-llm" in sys.argv
V = res / "v1"
K = json.loads((V / "frozen_knobs.json").read_text())
_emb_path = V / "emb_minilm_samples_v1.npy"
emb = np.load(_emb_path) if _emb_path.exists() else None
if emb is None:
    print("note: MiniLM embeddings not found (code/15_embed.py regenerates them); the weighted-CRC policy 'wcrc' is skipped", flush=True)
S = pd.read_parquet(data / "samples_v1.parquet", columns=["cid", "period", "set", "group", "month"])
ALPHAS = [0.05, 0.10, 0.15]
POLS = ["rolling_cod_known", "rolling_cod_none", "rolling_cod_det", "rolling_plugin_known", "rolling_plugin_none",
        "rolling_plugin_det", "static_cod", "static_plugin", "fullstatic_cod", "fullstatic_plugin", "crc", "wcrc", "aci"]
if emb is None:
    POLS = [p for p in POLS if p != "wcrc"]
BRK = {"A": "2017-04", "B": "2023-08"}
# classes new at THIS window's break only (fix 2026-11-02: the full dict also flagged pre-break
# 'Vehicle loan or lease' complaints in window B, a class created in 2017, as new)
NEWP = {"A": {"R1": NEW_CLASSES["R1"]}, "B": {"R2": NEW_CLASSES["R2"]}}


def kw():
    return dict(W=K["W"], lam0=K["lam0"], test=K["test"], gamma=K["aci_gamma"], emb=emb)


def stress_filter(per):
    tr_groups = set(S[(S["period"] == per) & S["set"].isin(["train", "audit_trn"])]["group"])
    bad = set(S[(S["period"] == per) & (S["set"] == "eval") & S["group"].isin(tr_groups)]["cid"])
    return lambda d: ~d["cid"].isin(bad)


def issue_new():
    sp = importlib.util.spec_from_file_location("mt", here / "14_monthly_tiers.py"); mt = importlib.util.module_from_spec(sp); sp.loader.exec_module(mt)
    I = mt.issue_sets(data)
    return {"R1": sorted(I["R1"] - I["R0"]), "R2": sorted(I["R2"] - I["R1"])}


out, summ_parts = [], []
tab = lambda f: pd.read_parquet(V / f)
# ---------------- Product A/B
t0 = tab("tier_tfidf_product.parquet")
for per in ["A", "B"]:
    t1 = tab(f"tier_mbert_product_{per}.parquet")
    M = h.merge_tiers({"t0": t0[t0["period"] == per], "t1": t1})
    for pol, a in itertools.product(POLS if not only_llm else [], ALPHAS):
        R = h.run(M, ["t0", "t1"], per, pol, a, new_classes=NEWP[per], **kw()); out.append(R.assign(task="product", split="main"))
    for a in (ALPHAS if not only_llm else []):
        out.append(h.run(M, ["t1"], per, "fullstatic_plugin", a, new_classes=NEWP[per], **kw()).assign(task="product", split="main", policy="B1_fatal"))
        out.append(h.run(M, ["t1"], per, "rolling_cod_known", a, new_classes=NEWP[per], **kw()).assign(task="product", split="main", policy="encoder_only_cod"))
        out.append(h.run(M, ["t0", "t1"], per, "rolling_cod_known", a, deploy_filter=stress_filter(per), new_classes=NEWP[per], **kw()).assign(task="product", split="stress"))
        out.append(h.run(M, ["t0", "t1"], per, "static_plugin", a, deploy_filter=stress_filter(per), new_classes=NEWP[per], **kw()).assign(task="product", split="stress"))
    if use_llm:
        t2 = tab(f"tier_llm_product_{per}.parquet")
        M3 = h.merge_tiers({"t0": t0[t0["period"] == per], "t1": t1, "t2": t2})
        months = sorted(M3["deploy_month"].unique())
        sub = lambda d: d["rank"] < 1000
        for a in ALPHAS:
            for tiers, lab in [(["t0", "t1", "t2"], "C4_with_llm"), (["t0", "t1"], "C4_without_llm")]:
                R = h.run(M3 if "t2" in tiers else M, tiers, per, "rolling_cod_known", a, months=months,
                          deploy_filter=sub, new_classes=NEWP[per], **kw())
                out.append(R.assign(task="product", split="main", policy=lab))
    print("product", per, "done", flush=True)
# ---------------- Issue A/D/B
ti, fi = tab("tier_tfidf_issue.parquet"), tab("tier_emb_issue.parquet")
inew_all = issue_new()
INEWP = {"A": {"R1": inew_all["R1"]}, "B": {"R2": inew_all["R2"]}, "D": {}}
for per in (["A", "D", "B"] if not only_llm else []):
    M = h.merge_tiers({"t0": ti[ti["period"] == per], "f": fi[fi["period"] == per]})
    for pol, a in itertools.product(POLS, ALPHAS):
        out.append(h.run(M, ["t0", "f"], per, pol, a, new_classes=INEWP[per], **kw()).assign(task="issue", split="main"))
    print("issue", per, "done", flush=True)
R = pd.concat(out, ignore_index=True)
if only_llm:
    R = pd.concat([pd.read_csv(V / "main_monthly.csv"), R], ignore_index=True)
R.to_csv(V / "main_monthly.csv", index=False)
S2 = []
for (task, split), g in R.groupby(["task", "split"]):
    s = h.summarize(g, breaks_month=BRK); s["task"], s["split"] = task, split; S2.append(s)
pd.concat(S2).to_csv(V / "main_summary.csv", index=False)
# ---------------- noise floor: audit resampling
rs = []
for per in (["A", "B"] if not only_llm else []):
    t1 = tab(f"tier_mbert_product_{per}.parquet")
    M = h.merge_tiers({"t0": t0[t0["period"] == per], "t1": t1})
    for pol in ["rolling_cod_known", "rolling_plugin_known", "crc"]:
        for seed in range(20):
            rs.append(h.run(M, ["t0", "t1"], per, pol, 0.10, resample_seed=seed, **kw()).assign(seed=seed))
if rs:
    pd.concat(rs).to_csv(V / "resample_monthly.csv", index=False)
print("ALL DONE")
