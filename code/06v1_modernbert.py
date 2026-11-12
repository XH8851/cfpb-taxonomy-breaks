"""SPEC v1 tier T1: ModernBERT-base, 4 fine-tunes (per window A/B: m_start at eval start on the
train pool; m_re at break+3 on the train pool mapped into the new regime + audit_trn of months
<= break+2), bf16, len 256, 30k items, 1 epoch. Output is the long tier format used by
16_harness.py, keyed by (period, deploy_month, model_regime, role, cid), plus 'static' rows
(m_start on every eval/audit/preaudit item = fatal baseline B1).

Model deployed in month m: m_start if m <= break+2 else m_re.
Rows per (m, r): deploy = eval items of m in regime r; cal = audit_cal of m-6..m-1 with regime <= r
(label mapped into r); grid = eval items of m-1 in regime r.

Usage: python -I -u scripts/06v1_modernbert.py <data_dir> <results_dir> [A|B]
Output: results/v1/tier_mbert_product_<P>.parquet; weights are never written to disk.
"""
import sys, json, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
sys.path.insert(0, str(Path(__file__).parent))
from labels import LABELS, map_label

MODEL = "answerdotai/ModernBERT-base"
MAXLEN, BS, ACCUM, LR, N_TRAIN = 256, 16, 2, 5e-5, 30_000
BREAK = {"A": "2017-04", "B": "2023-08"}
OLD, NEW = {"A": "R0", "B": "R1"}, {"A": "R1", "B": "R2"}
ORDER = ["R0", "R1", "R2"]
DEV = torch.device("mps")


def mshift(m, k):
    return str(pd.Period(m, "M") + k)


def lab_in(y, s, r):
    if s == r:
        return y
    return map_label(y, s, r) if ORDER.index(s) < ORDER.index(r) else None


def train(texts, labels, classes, seed=13):
    torch.manual_seed(seed); np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(classes), attn_implementation="sdpa").to(DEV)
    y = np.array([classes.index(l) for l in labels])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = math.ceil(len(texts) / (BS * ACCUM))
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    model.train(); t0 = time.time()
    perm = np.random.permutation(len(texts))
    for j, i in enumerate(range(0, len(perm), BS)):
        idx = perm[i:i + BS]
        b = tok([texts[k] for k in idx], truncation=True, max_length=MAXLEN, padding="max_length", return_tensors="pt").to(DEV)
        with torch.autocast("mps", dtype=torch.bfloat16):
            out = model(**b, labels=torch.tensor(y[idx], device=DEV))
        (out.loss / ACCUM).backward()
        if j % ACCUM == ACCUM - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad()
        if j % 200 == 0:
            print(f"  batch {j}/{len(perm)//BS} loss {out.loss.item():.3f} t={time.time()-t0:.0f}s", flush=True)
    return tok, model, time.time() - t0


@torch.no_grad()
def predict(tok, model, texts):
    model.eval(); P = []
    for i in range(0, len(texts), 32):
        b = tok(texts[i:i + 32], truncation=True, max_length=MAXLEN, padding="max_length", return_tensors="pt").to(DEV)
        with torch.autocast("mps", dtype=torch.bfloat16):
            lg = model(**b).logits
        P.append(torch.softmax(lg.float(), -1).cpu().numpy())
        if (i // 32) % 500 == 0:
            print(f"  predict {i}/{len(texts)}", flush=True)
    return np.concatenate(P)


def mapped(P, classes, src, r):
    tgt = LABELS[r]
    M = np.zeros((len(classes), len(tgt)))
    for i, c in enumerate(classes):
        m = c if r == src else map_label(c, src, r)
        if m is not None:
            M[i, tgt.index(m)] = 1.0
    Q = P @ M
    return np.array(tgt, dtype=object)[Q.argmax(1)], Q.max(1)


def build_rows(S, months, r_of_month):
    """Key rows (deploy/cal/grid) for the given deploy months."""
    ev = S[S["set"] == "eval"]
    out = []
    for m in months:
        for r in sorted(ev.loc[ev["month"] == m, "regime"].unique()):
            cal = S[(S["set"] == "audit_cal") & S["month"].isin([mshift(m, -k) for k in range(1, 7)])]
            cal = cal[[ORDER.index(x) <= ORDER.index(r) for x in cal["regime"]]]
            for role, d in [("deploy", ev[(ev["month"] == m) & (ev["regime"] == r)]), ("cal", cal),
                            ("grid", ev[(ev["month"] == mshift(m, -1)) & (ev["regime"] == r)])]:
                out.append(d.assign(role=role, deploy_month=m, model_regime=r))
    return pd.concat(out, ignore_index=True)


def main(data, res, per):
    S = pd.read_parquet(data / "samples_v1.parquet")
    S["eidx"] = np.arange(len(S))
    S = S[S["period"] == per].copy()  # (other periods are dropped right away to save memory)
    import gc; gc.collect()
    bm = BREAK[per]
    months = sorted(S.loc[S["set"] == "eval", "month"].unique())
    m_cut = mshift(bm, 3)
    parts, meta = [], {}
    # --- m_start
    tr = S[(S["set"] == "train") & (S["regime"] == OLD[per])].sample(frac=1, random_state=13).head(N_TRAIN)
    classes = sorted(tr["product"].unique())
    tok, model, sec = train(tr["text"].tolist(), tr["product"].tolist(), classes)
    rows = build_rows(S, [m for m in months if m < m_cut], None)
    static = S[S["set"].isin(["eval", "audit_cal", "audit_trn", "preaudit"])].copy()
    need = pd.concat([rows[["cid", "text"]], static[["cid", "text"]]]).drop_duplicates("cid")
    P = predict(tok, model, need["text"].tolist())
    pos = pd.Series(np.arange(len(need)), index=need["cid"])
    for r in rows["model_regime"].unique():
        g = rows[rows["model_regime"] == r].copy()
        g["pred"], g["score"] = mapped(P[pos.loc[g["cid"]].values], classes, OLD[per], r)
        parts.append(g)
    for r in static["regime"].unique():
        g = static[static["regime"] == r].copy()
        g["pred"], g["score"] = mapped(P[pos.loc[g["cid"]].values], classes, OLD[per], r)
        parts.append(g.assign(role="static", deploy_month="static", model_regime=r))
    meta["m_start"] = {"n_train": len(tr), "classes": classes, "train_s": round(sec), "n_scored": len(need)}
    del model; torch.mps.empty_cache()
    print(json.dumps(meta["m_start"])[:300], flush=True)
    # --- m_re (break+3)
    pool = S[(S["set"] == "train") | ((S["set"] == "audit_trn") & (S["month"] <= mshift(bm, 2)))].copy()
    pool["lab"] = [lab_in(y, s, NEW[per]) for y, s in zip(pool["product"], pool["regime"])]
    pool = pool[pool["lab"].notna()]
    post = pool[pool["set"] == "audit_trn"]
    tr = pd.concat([post, pool[pool["set"] == "train"].sample(frac=1, random_state=13).head(N_TRAIN - len(post))])
    classes = sorted(tr["lab"].unique())
    tok, model, sec = train(tr["text"].tolist(), tr["lab"].tolist(), classes)
    rows = build_rows(S, [m for m in months if m >= m_cut], None)
    need = rows[["cid", "text"]].drop_duplicates("cid")
    P = predict(tok, model, need["text"].tolist())
    pos = pd.Series(np.arange(len(need)), index=need["cid"])
    for r in rows["model_regime"].unique():
        g = rows[rows["model_regime"] == r].copy()
        g["pred"], g["score"] = mapped(P[pos.loc[g["cid"]].values], classes, NEW[per], r)
        parts.append(g)
    meta["m_re"] = {"n_train": len(tr), "n_post_break_labels": len(post), "classes": classes,
                    "train_s": round(sec), "n_scored": len(need)}
    del model; torch.mps.empty_cache()
    df = pd.concat(parts, ignore_index=True)
    df["label"] = df["product"]
    df["label_r"] = [lab_in(y, s, r) for y, s, r in zip(df["product"], df["regime"], df["model_regime"])]
    df["correct"] = df["pred"] == df["label_r"]
    df["tier"], df["task"] = "t1", "product"
    cols = ["period", "deploy_month", "model_regime", "role", "cid", "set", "month", "regime", "label", "label_r",
            "rank", "eidx", "pred", "score", "correct", "tier", "task"]
    df[cols].to_parquet(res / "v1" / f"tier_mbert_product_{per}.parquet", index=False)
    (res / "v1" / f"mbert_meta_{per}.json").write_text(json.dumps(meta, indent=1))
    print("done", per, len(df))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
