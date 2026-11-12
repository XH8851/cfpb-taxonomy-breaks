"""SPEC v1 LLM tier: build (a) the items to score and (b) the few-shot retrieval pool, and (c)
expand per-item LLM scores into harness rows.

Scope: windows A and B, deploy months break-2 .. break+6; eval items with rank < 1000;
calibration items = audit_cal of months m-W..m-1 in the item's regime (reset@known only);
grid items = eval rank < 1000 of month m-1.
Few-shot pool (label economy): train pool + audit_trn of the same period; 08_llm_tier.py only
uses pool items of the same regime from months strictly before the scored item's month.
Calibration labels (audit_cal) are never used as examples.

Usage:
  python -I scripts/08v1_llm_prep.py prep <data_dir> <results_dir> <W>
  python -I scripts/08v1_llm_prep.py expand <data_dir> <results_dir> <W>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BREAK = {"A": "2017-04", "B": "2023-08"}


def mshift(m, k):
    return str(pd.Period(m, "M") + k)


def rows_for(S, per, W):
    bm = BREAK[per]
    ev = S[(S["set"] == "eval") & (S["rank"] < 1000)]
    out = []
    for m in [mshift(bm, k) for k in range(-2, 7)]:
        for r in sorted(ev.loc[ev["month"] == m, "regime"].unique()):
            cal = S[(S["set"] == "audit_cal") & S["month"].isin([mshift(m, -k) for k in range(1, W + 1)]) & (S["regime"] == r)]
            for role, d in [("deploy", ev[(ev["month"] == m) & (ev["regime"] == r)]), ("cal", cal),
                            ("grid", ev[(ev["month"] == mshift(m, -1)) & (ev["regime"] == r)])]:
                out.append(d.assign(role=role, deploy_month=m, model_regime=r))
    return pd.concat(out, ignore_index=True)


def main():
    mode, data, res, W = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
    S = pd.read_parquet(data / "samples_v1.parquet")
    S["eidx"] = np.arange(len(S))
    for per in ["A", "B"]:
        Sp = S[S["period"] == per]
        R = rows_for(Sp, per, W)
        if mode == "prep":
            items = R.drop_duplicates("cid")[["cid", "month", "regime", "product", "text"]]
            items.to_parquet(res / "v1" / f"llm_items_{per}.parquet", index=False)
            pool = Sp[Sp["set"].isin(["train", "audit_trn"])][["cid", "month", "regime", "product", "text"]]
            pool.to_parquet(res / "v1" / f"llm_pool_{per}.parquet", index=False)
            print(per, "rows", len(R), "distinct items", len(items), "pool", len(pool))
        else:
            sc = pd.read_parquet(res / "v1" / f"llm_scores_{per}.parquet")[["cid", "pred", "score"]]
            d = R.merge(sc, on="cid", how="inner")
            d["label"] = d["product"]; d["label_r"] = d["product"]
            d["correct"] = d["pred"] == d["label_r"]
            d["tier"], d["task"] = "t2", "product"
            cols = ["period", "deploy_month", "model_regime", "role", "cid", "set", "month", "regime", "label",
                    "label_r", "rank", "eidx", "pred", "score", "correct", "tier", "task"]
            d[cols].to_parquet(res / "v1" / f"tier_llm_product_{per}.parquet", index=False)
            print(per, "rows", len(d))


if __name__ == "__main__":
    main()
