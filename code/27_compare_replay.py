"""Compare a fresh replay (results/v1/main_monthly.csv) with the shipped copy
(results/v1_shipped/main_monthly.csv) on errors and automation, row by row.
Usage: python -I code/27_compare_replay.py <results_dir>"""
import sys
from pathlib import Path
import pandas as pd
res = Path(sys.argv[1])
new, old = pd.read_csv(res / "v1" / "main_monthly.csv"), pd.read_csv(res / "v1_shipped" / "main_monthly.csv")
k = ["task", "split", "period", "policy", "alpha", "month"]
m = old.merge(new, on=k, suffixes=("_shipped", "_replay"))
same = (m["n_err_shipped"] == m["n_err_replay"]) & (m["n_auto_shipped"] == m["n_auto_replay"])
print(f"compared {len(m)} monthly rows (policies replayed: {m['policy'].nunique()}); identical errors and automation: {same.mean():.4%}")
missing = set(map(tuple, old[k].values)) - set(map(tuple, new[k].values))
print(f"shipped rows not replayed: {len(missing)} (e.g. weighted CRC when embeddings are absent, or rows from --only-llm)")
sys.exit(0 if same.all() else 1)
