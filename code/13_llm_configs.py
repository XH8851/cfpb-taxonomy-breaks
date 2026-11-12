"""Phase 2: compare LLM-tier configurations on the 1,883 pilot items and apply SPEC v1 §4.1.
Configs: single (forward), avg2 (forward+reverse), avg3 (+shuffle), fewshot (k=4, forward).
Reports accuracy (natural / new class), selective accuracy at 30/50/70% coverage, share of
scores >= 0.999, ECE (10 bins), cost multiplier. Usage: python -I scripts/13_llm_configs.py <results_dir>"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

res = Path(sys.argv[1])
I = pd.read_parquet(res / "llm_items.parquet")[["cid", "window", "pick", "product"]]


def load(name):
    d = pd.read_parquet(res / f"llm_q4b_{name}.parquet")
    d["P"] = d["probs"].map(json.loads)
    return d.set_index("cid")


raw = {k: load(k) for k in ["main", "reverse", "shuffle", "fewshot"]}
sec = {k: float(v["sec"].median()) for k, v in raw.items()}


def avg(names):
    base = raw[names[0]]
    rows = []
    for cid in base.index:
        ps = [raw[n].at[cid, "P"] for n in names]
        labs = ps[0].keys()
        m = {l: float(np.mean([p[l] for p in ps])) for l in labs}
        k = max(m, key=m.get)
        rows.append((cid, k, m[k]))
    return pd.DataFrame(rows, columns=["cid", "pred", "score"]).set_index("cid")


cfg = {"single": raw["main"][["pred", "score"]], "avg2": avg(["main", "reverse"]),
       "avg3": avg(["main", "reverse", "shuffle"]), "fewshot": raw["fewshot"][["pred", "score"]]}
cost = {"single": 1.0, "avg2": 2.0, "avg3": 3.0, "fewshot": sec["fewshot"] / sec["main"]}


def ece(s, c, bins=10):
    e = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(s, e) - 1, 0, bins - 1)
    return float(sum(abs(c[idx == b].mean() - s[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


out = {}
for name, d in cfg.items():
    j = d.join(I.set_index("cid"))
    j["correct"] = j["pred"] == j["product"]
    r = {"cost_x": round(cost[name], 2)}
    for w in "AB":
        g = j[j["window"] == w]
        nat, new = g[g["pick"] == "natural"], g[g["pick"] == "new_class"]
        s = nat.sort_values("score", ascending=False)
        r[w] = {"acc_natural": round(nat["correct"].mean(), 3), "acc_new": round(new["correct"].mean(), 3),
                **{f"sel@{c}": round(s.head(int(len(s) * c))["correct"].mean(), 3) for c in (0.3, 0.5, 0.7)},
                "share_ge_0.999": round(float((nat["score"] >= 0.999).mean()), 3),
                "ece": round(ece(nat["score"].values, nat["correct"].values.astype(float)), 3)}
    r["sel@0.5_mean"] = round((r["A"]["sel@0.5"] + r["B"]["sel@0.5"]) / 2, 4)
    out[name] = r
flip = {f"main_vs_{k}": round(float((raw["main"]["pred"] != raw[k].loc[raw["main"].index, "pred"]).mean()), 3)
        for k in ["reverse", "shuffle", "fewshot"]}
eligible = {k: v for k, v in out.items() if v["cost_x"] <= 2.0}
choice = max(eligible, key=lambda k: eligible[k]["sel@0.5_mean"])
out["_flip_rates"] = flip
out["_sec_p50"] = sec
out["_rule"] = "SPEC v1 §4.1: highest mean selective accuracy at 50% coverage (natural items, A and B) among configs with cost <= 2x"
out["_choice"] = choice
(res / "pilot" / "llm_configs.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
