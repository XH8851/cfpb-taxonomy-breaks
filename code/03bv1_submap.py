"""SPEC v1 LLM option descriptions: sub-product names observed in the FIRST 30 DAYS of each
regime (what the CFPB form offered when the regime started; v0 used the whole period, which
leaked later information). Names seen fewer than 3 times in that span are dropped.
Usage: python -I scripts/03bv1_submap.py <data_dir>  ->  data/submap_v1.json"""
import sys, json
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from labels import LABELS

START = {"R0": "2015-03-19", "R1": "2017-04-24", "R2": "2023-08-25"}
data = Path(sys.argv[1])
df = pd.read_parquet(data / "narratives.parquet", columns=["date", "product", "sub_product"])
out = {}
for r, s in START.items():
    e = str((pd.Timestamp(s) + pd.Timedelta(days=29)).date())
    d = df[(df["date"] >= s) & (df["date"] <= e) & (df["sub_product"].str.strip() != "")]
    c = d.groupby(["product", "sub_product"]).size()
    out[r] = {p: sorted(sp for (pp, sp), n in c.items() if pp == p and n >= 3) for p in LABELS[r]}
(data / "submap_v1.json").write_text(json.dumps(out, indent=1))
for r in out:
    print(r, {p: len(v) for p, v in out[r].items()})
