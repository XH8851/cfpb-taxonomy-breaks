"""SPEC v1 §4.1 LLM drop rule and new-class breakdown for C4 (break..break+6, rank<1000 stream).
Rule: drop the LLM if it adds < 2 points of certified automation in at least half of the months.
Usage: python -I scripts/19b_c4_drop_rule.py <results_dir>  ->  results/v1/c4_drop_rule.json"""
import sys, json
from pathlib import Path
import pandas as pd
res = Path(sys.argv[1]); R = pd.read_csv(res / "v1" / "main_monthly.csv")
BRK = {"A": "2017-04", "B": "2023-08"}; ms = lambda m, k: str(pd.Period(m, "M") + k)
out = {}
for a in [0.10, 0.15, 0.05]:
    rows = []
    for per in "AB":
        w = R[(R.policy == "C4_with_llm") & (R.alpha == a) & (R.period == per)].set_index("month")
        o = R[(R.policy == "C4_without_llm") & (R.alpha == a) & (R.period == per)].set_index("month")
        for m in [ms(BRK[per], k) for k in range(7)]:
            rows.append(dict(period=per, month=m, gain_pts=100 * (w.at[m, "automation"] - o.at[m, "automation"]),
                             t2_share=100 * w.at[m, "auto_t2"], new=int(w.at[m, "n_new"]),
                             ok_with=int(w.at[m, "new_correct_auto"]), ok_without=int(o.at[m, "new_correct_auto"]),
                             bad_with=int(w.at[m, "new_wrong_auto"]), bad_without=int(o.at[m, "new_wrong_auto"])))
    T = pd.DataFrame(rows)
    out[str(a)] = {"months_gain_ge_2pts": int((T.gain_pts >= 2).sum()), "of": len(T),
                   "drop_rule_fires": bool((T.gain_pts >= 2).mean() < 0.5), "mean_gain_pts": round(T.gain_pts.mean(), 2),
                   "mean_t2_share_pts": round(T.t2_share.mean(), 2), "new_items": int(T.new.sum()),
                   "new_correct_auto_with": int(T.ok_with.sum()), "new_correct_auto_without": int(T.ok_without.sum()),
                   "new_wrong_auto_with": int(T.bad_with.sum()), "new_wrong_auto_without": int(T.bad_without.sum())}
(res / "v1" / "c4_drop_rule.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["0.1"], indent=1))
