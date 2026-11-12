"""SPEC v1 §5.3 (Issue task, Amendment v1.3): freeze CoD knobs on the Issue task; (W, lam0, test) and ACI gamma on the dev period D only.

Cascade on D: T0 -> F -> human (Product). Placebo dates 2019-10-01 and 2020-10-01 act as the
"known" resets. Rule (pre-registered): maximize mean automation at alpha = 0.10 subject to
excess errors per 1k <= 1.0 overall AND within +-3 months of each placebo date.
Also writes the full dev-period factorial (reportable as dev results).

Usage: python -I scripts/17_freeze_dev.py <results_dir>
Output: results/v1/frozen_knobs_issue.json, results/v1/dev_knob_grid_issue.csv, results/v1/dev_factorial_*.csv
"""
import sys, json, itertools, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

here = Path(__file__).parent
spec = importlib.util.spec_from_file_location("h", here / "16_harness.py"); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
res = Path(sys.argv[1])
T0 = pd.read_parquet(res / "v1" / "tier_tfidf_issue.parquet")
F = pd.read_parquet(res / "v1" / "tier_emb_issue.parquet")
M = h.merge_tiers({"t0": T0[T0["period"] == "D"], "f": F[F["period"] == "D"]})
_emb_path = res / "v1" / "emb_minilm_samples_v1.npy"
emb = np.load(_emb_path) if _emb_path.exists() else None
if emb is None:
    print("note: MiniLM embeddings not found (code/15_embed.py regenerates them); the weighted-CRC policy 'wcrc' is skipped", flush=True)
tiers = ["t0", "f"]
PLAC = ["2019-10", "2020-10"]


def excess_near(R):
    out = []
    for p in PLAC:
        g = R[(R["month"] >= h.mshift(p, -3)) & (R["month"] <= h.mshift(p, 3))]
        out.append(1000 * g["excess"].sum() / g["n"].sum())
    return max(out)


rows = []
for W, lam0, test in itertools.product([3, 6], [0.10, 0.30], ["fs", "ms"]):
    R = h.run(M, tiers, "D", "rolling_cod_known", 0.10, W=W, lam0=lam0, test=test)
    rows.append({"W": W, "lam0": lam0, "test": test, "automation": R["automation"].mean(),
                 "excess_per_1k": 1000 * R["excess"].sum() / R["n"].sum(), "excess_near_placebo": excess_near(R),
                 "ci_lo_over": int((R.loc[~R["vacuous"], "ci_lo"] > 0.10).sum()), "vacuous": int(R["vacuous"].sum())})
    print(rows[-1], flush=True)
G = pd.DataFrame(rows)
G.to_csv(res / "v1" / "dev_knob_grid_issue.csv", index=False)
ok = G[(G["excess_per_1k"] <= 1.0) & (G["excess_near_placebo"] <= 1.0)]
pick = (ok if len(ok) else G.assign(_p=G["excess_per_1k"])).sort_values(
    ["automation"] if len(ok) else ["_p"], ascending=[False] if len(ok) else [True]).iloc[0]
knobs = {"W": int(pick["W"]), "lam0": float(pick["lam0"]), "test": pick["test"],
         "rule_satisfied": bool(len(ok) > 0)}
grows = []
for g in [0.005, 0.01, 0.05]:
    R = h.run(M, tiers, "D", "aci", 0.10, W=knobs["W"], lam0=knobs["lam0"], gamma=g)
    grows.append({"gamma": g, "automation": R["automation"].mean(), "excess_per_1k": 1000 * R["excess"].sum() / R["n"].sum(),
                  "excess_near_placebo": excess_near(R)})
GA = pd.DataFrame(grows)
okA = GA[(GA["excess_per_1k"] <= 1.0) & (GA["excess_near_placebo"] <= 1.0)]
knobs["aci_gamma"] = float((okA.sort_values("automation", ascending=False) if len(okA) else GA.sort_values("excess_per_1k")).iloc[0]["gamma"])
knobs["aci_grid"] = GA.round(4).to_dict(orient="records")
(res / "v1" / "frozen_knobs_issue.json").write_text(json.dumps(knobs, indent=1))
print("FROZEN", knobs)
# full dev factorial with frozen knobs (reportable dev results)
pols = ["rolling_cod_known", "rolling_cod_none", "rolling_cod_det", "rolling_plugin_known", "rolling_plugin_none",
        "rolling_plugin_det", "static_cod", "static_plugin", "fullstatic_cod", "fullstatic_plugin", "crc", "wcrc", "aci"]
if emb is None:
    pols = [p for p in pols if p != "wcrc"]
allR = []
for pol, a in itertools.product(pols, [0.05, 0.10, 0.15]):
    allR.append(h.run(M, tiers, "D", pol, a, W=knobs["W"], lam0=knobs["lam0"], test=knobs["test"],
                      gamma=knobs["aci_gamma"], emb=emb))
R = pd.concat(allR)
R.to_csv(res / "v1" / "dev_factorial_issue_monthly.csv", index=False)
S = h.summarize(R)
S.to_csv(res / "v1" / "dev_factorial_issue_summary.csv", index=False)
pd.set_option("display.width", 220)
print(S[["policy", "alpha", "excess_per_1k", "ci_lo_over", "ci_lo_expected_nominal", "vacuous_months", "automation"]].to_string(index=False))
