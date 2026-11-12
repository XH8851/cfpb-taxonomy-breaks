"""Contamination probe items (SPEC v1 §4.1): 1,000 R2 narratives received 2023-08-25..2024-07-31
(before Qwen3-2507's release) and 1,000 received 2025-09-01..2026-08-11 (after it), uniform.
Usage: python -I scripts/08c_contam_items.py <data_dir> <results_dir>"""
import sys
from pathlib import Path
import pandas as pd
data, res = Path(sys.argv[1]), Path(sys.argv[2])
df = pd.read_parquet(data / "narratives.parquet", columns=["cid", "date", "month", "product", "text"])
df = df[df["date"] >= "2023-08-25"]
old = df[df["date"] <= "2024-07-31"].sample(1000, random_state=31).assign(era="2023-08..2024-07")
new = df[df["date"] >= "2025-09-01"].sample(1000, random_state=31).assign(era="2025-09..2026-08")
out = pd.concat([old, new]).assign(regime="R2")
out.to_parquet(res / "v1" / "contam_items.parquet", index=False)
print(out.groupby("era").size().to_dict(), out.groupby("era")["product"].agg(lambda s: s.value_counts(normalize=True).iloc[0]).round(3).to_dict())
