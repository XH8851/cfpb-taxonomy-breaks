"""Re-run the Issue-task replays (A, D, B) with Issue-specific knobs frozen on D
(results/v1/frozen_knobs_issue.json, SPEC Amendment v1.3); replaces the Issue rows of
main_monthly.csv / main_summary.csv. Usage: python -I scripts/18b_replay_issue.py <data_dir> <results_dir>"""
import sys, json, itertools, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
here = Path(__file__).parent
spec = importlib.util.spec_from_file_location("h", here / "16_harness.py"); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
spec = importlib.util.spec_from_file_location("mt", here / "14_monthly_tiers.py"); mt = importlib.util.module_from_spec(spec); spec.loader.exec_module(mt)
data, res = Path(sys.argv[1]), Path(sys.argv[2]); V = res / "v1"
K = json.loads((V / "frozen_knobs_issue.json").read_text())
_emb_path = V / "emb_minilm_samples_v1.npy"
emb = np.load(_emb_path) if _emb_path.exists() else None
if emb is None:
    print("note: MiniLM embeddings not found (code/15_embed.py regenerates them); the weighted-CRC policy 'wcrc' is skipped", flush=True)
I = mt.issue_sets(data)
INEWP = {"A": {"R1": sorted(I["R1"] - I["R0"])}, "B": {"R2": sorted(I["R2"] - I["R1"])}, "D": {}}  # new at this window's break only
POLS = ["rolling_cod_known", "rolling_cod_none", "rolling_cod_det", "rolling_plugin_known", "rolling_plugin_none",
        "rolling_plugin_det", "static_cod", "static_plugin", "fullstatic_cod", "fullstatic_plugin", "crc", "wcrc", "aci"]
if emb is None:
    POLS = [p for p in POLS if p != "wcrc"]
ti, fi = pd.read_parquet(V / "tier_tfidf_issue.parquet"), pd.read_parquet(V / "tier_emb_issue.parquet")
out = []
for per in ["A", "D", "B"]:
    M = h.merge_tiers({"t0": ti[ti["period"] == per], "f": fi[fi["period"] == per]})
    for pol, a in itertools.product(POLS, [0.05, 0.10, 0.15]):
        out.append(h.run(M, ["t0", "f"], per, pol, a, W=K["W"], lam0=K["lam0"], test=K["test"], gamma=K["aci_gamma"],
                         emb=emb, new_classes=INEWP[per]).assign(task="issue", split="main"))
    print("issue", per, flush=True)
R = pd.read_csv(V / "main_monthly.csv")
R = pd.concat([R[R["task"] != "issue"], pd.concat(out)], ignore_index=True)
R.to_csv(V / "main_monthly.csv", index=False)
BRK = {"A": "2017-04", "B": "2023-08"}
S2 = []
for (task, split), g in R.groupby(["task", "split"]):
    s = h.summarize(g, breaks_month=BRK); s["task"], s["split"] = task, split; S2.append(s)
pd.concat(S2).to_csv(V / "main_summary.csv", index=False)
print("done")
