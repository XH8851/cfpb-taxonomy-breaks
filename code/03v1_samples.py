"""SPEC v1 samples for periods A, D (dev) and B under the v1 label economy.

Per period:
  train      up to 5,000 items/month in the 12 months before eval start (labeled history)
  preaudit   300 items/month in the last 3 pre-eval months (static calibration); disjoint from train
  audit_cal  300 items/month in eval months (calibration only)
  audit_trn  300 items/month in eval months (retraining only)
  eval       2,000 items/month in eval months; `rank` 0..1999 is a fixed random order and LLM
             configurations use rank < 1000
All draws are uniform within the month (item level). `group` (MinHash near-duplicate group from
02v1) is kept for the group-disjoint stress split.

Usage: python -I scripts/03v1_samples.py <data_dir>
Output: data/samples_v1.parquet, results/samples_v1_summary.json
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from labels import LABELS, regime_of

SEED = 20261101
PERIODS = {
    "A": {"train": ("2015-04", "2016-03"), "preaudit": ("2016-01", "2016-03"), "eval": ("2016-04", "2018-12")},
    "D": {"train": ("2018-01", "2018-12"), "preaudit": ("2018-10", "2018-12"), "eval": ("2019-01", "2021-07")},
    "B": {"train": ("2021-08", "2022-07"), "preaudit": ("2022-05", "2022-07"), "eval": ("2022-08", "2025-02")},
}


def inr(m, r):
    return (m >= r[0]) & (m <= r[1])


def main(data: Path) -> None:
    w = pd.read_parquet(data / "window_items_v1.parquet",
                        columns=["cid", "date", "product", "issue", "text", "group", "group_size"])
    w["month"] = w["date"].str[:7]
    w["regime"] = regime_of(w["date"])
    ok = np.array([p in LABELS[r] for p, r in zip(w["product"], w["regime"])])
    dropped = int((~ok).sum())
    w = w[ok].sample(frac=1.0, random_state=SEED).reset_index(drop=True)  # shuffled once
    out = []
    for per, R in PERIODS.items():
        # pre-eval months: preaudit first (300), then train (<=5000) from the remaining items
        pre = w[inr(w["month"], R["train"])].copy()
        pre["k"] = pre.groupby("month").cumcount()
        is_pa = inr(pre["month"], R["preaudit"]) & (pre["k"] < 300)
        pa = pre[is_pa].assign(set="preaudit")
        rest = pre[~is_pa].copy()
        rest["k"] = rest.groupby("month").cumcount()
        tr = rest[rest["k"] < 5000].assign(set="train")
        ev = w[inr(w["month"], R["eval"])].copy()
        ev["k"] = ev.groupby("month").cumcount()
        au = ev[ev["k"] < 600].copy()
        au["set"] = np.where(au["k"] % 2 == 0, "audit_cal", "audit_trn")
        es = ev[(ev["k"] >= 600) & (ev["k"] < 2600)].copy()
        es["set"], es["rank"] = "eval", es["k"] - 600
        for part in (pa, tr, au, es):
            out.append(part.assign(period=per))
    df = pd.concat(out, ignore_index=True).drop(columns=["k"])
    df["rank"] = df["rank"].fillna(-1).astype(int)
    df.to_parquet(data / "samples_v1.parquet", compression="zstd", index=False)
    summ = (df.groupby(["period", "set"])
              .agg(n=("cid", "size"), months=("month", "nunique"),
                   min_per_month=("month", lambda m: int(m.value_counts().min())))
              .reset_index())
    print("dropped (label not in date regime):", dropped)
    print(summ.to_string(index=False))
    (data.parent / "results" / "samples_v1_summary.json").write_text(
        json.dumps({"dropped": dropped, "seed": SEED, "summary": summ.to_dict(orient="records")}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
