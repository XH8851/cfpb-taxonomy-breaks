"""Review round 4 (2026-11-13, post-hoc, disclosed): (1) SGR baseline; (2) risk-automation frontier
over nominal levels, for a comparison at matched automation; (3) inputs for the clipping noise floor.
Product routing, cascade T0 -> T1, windows A and B, frozen knobs (W, lam0, test) from frozen_knobs.json.
Runs rolling_{cod,plugin,sgr}_known and crc at nominal levels LEVELS. First checks that the edited
harness reproduces the shipped rolling_cod_known, rolling_plugin_known and crc rows at alpha = 0.10.
Usage: python -I scripts/32_frontier_sgr.py <data_dir> <results_dir>
Output: results/v1/frontier_monthly.csv
"""
import sys, json, time, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

here = Path(__file__).parent
spec = importlib.util.spec_from_file_location("h", here / "16_harness.py"); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
sys.path.insert(0, str(here))
from labels import NEW_CLASSES

data, res = Path(sys.argv[1]), Path(sys.argv[2]); V = res / "v1"
K = json.loads((V / "frozen_knobs.json").read_text())
kw = dict(W=K["W"], lam0=K["lam0"], test=K["test"], gamma=K["aci_gamma"])
NEWP = {"A": {"R1": NEW_CLASSES["R1"]}, "B": {"R2": NEW_CLASSES["R2"]}}
LEVELS = [round(x, 4) for x in np.arange(0.03, 0.1201, 0.005)]
POLS = ["rolling_cod_known", "rolling_plugin_known", "crc", "rolling_sgr_known"]
ship = pd.read_csv(V / "main_monthly.csv")
t0 = pd.read_parquet(V / "tier_tfidf_product.parquet")
out = []
for per in ["A", "B"]:
    t1 = pd.read_parquet(V / f"tier_mbert_product_{per}.parquet")
    M = h.merge_tiers({"t0": t0[t0["period"] == per], "t1": t1})
    for pol0 in ["rolling_cod_known", "rolling_plugin_known", "crc"]:
        chk = h.run(M, ["t0", "t1"], per, pol0, 0.10, new_classes=NEWP[per], **kw)
        s = ship[(ship["task"] == "product") & (ship["split"] == "main") & (ship["period"] == per) & (ship["policy"] == pol0) & (ship["alpha"] == 0.10)]
        j = chk.merge(s, on="month", suffixes=("", "_s"))
        assert len(j) == len(s) and (j["n_err"] == j["n_err_s"]).all() and (j["n_auto"] == j["n_auto_s"]).all(), f"harness edit changed {pol0}"
        print(f"{per}: edited harness reproduces shipped {pol0} ({len(j)} months)", flush=True)
    for pol in POLS:
        t = time.time()
        for a in LEVELS:
            out.append(h.run(M, ["t0", "t1"], per, pol, a, new_classes=NEWP[per], **kw))
        print(f"{per} {pol}: {len(LEVELS)} levels in {time.time() - t:.0f}s", flush=True)
pd.concat(out).to_csv(V / "frontier_monthly.csv", index=False)
print("wrote", V / "frontier_monthly.csv")
