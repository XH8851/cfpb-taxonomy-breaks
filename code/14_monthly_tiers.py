"""SPEC v1 monthly-retrained tiers (T0 = TF-IDF+LR, F = frozen MiniLM embedding + LR) for
Product and Issue, plus the static (eval-start) model of each.

For eval month m and regime r present in month m, model(m, r) is trained on
  train pool (pre-eval history) + audit_trn items with month <= m-1 (label delay d=1),
with labels mapped into regime r (Product: labels.MAP carry-over edges; Issue: name identity;
items without carry-over dropped). It scores, in regime r:
  eval items of month m            (deployment)
  audit_cal items of months m-6..m-1 (calibration window, W <= 6)
  eval items of month m-1          (unlabeled grid source)
The static model (trained at eval start on the train pool, regime of the last pre-eval month)
scores every eval/audit/preaudit item of the period, mapped into the item's regime.

Usage: python -I scripts/14_monthly_tiers.py <data_dir> <results_dir> <tfidf|emb> <product|issue> [n_jobs]
Output: results/v1/tier_<feat>_<task>.parquet (long: one row per item per deploy month)
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).parent))
from labels import LABELS, map_label

ORDER = ["R0", "R1", "R2"]
MIN_CLASS = 50  # Issue classes need >= 50 narratives in the regime


def mshift(m, k):
    return str(pd.Period(m, "M") + k)


def issue_sets(data):
    C = pd.read_parquet(data / "label_counts_narr.parquet")
    C["regime"] = np.where(C["month"] < "2017-05", "R0", np.where(C["month"] < "2023-09", "R1", "R2"))
    # the two break months mix regimes; restrict to months fully inside a regime
    C = C[~C["month"].isin(["2017-04", "2023-08"])]
    g = C.groupby(["regime", "issue"])["n"].sum()
    return {r: set(i for (rr, i), n in g.items() if rr == r and n >= MIN_CLASS) for r in ORDER}


def to_regime(labels, src_regimes, r, task, isets):
    out = []
    for y, s in zip(labels, src_regimes):
        if task == "product":
            out.append(y if s == r else (map_label(y, s, r) if ORDER.index(s) < ORDER.index(r) else None))
        else:
            out.append(y if (y in isets[r]) else None)
    return out


def featurize(feat, train_text, test_text, emb, train_idx, test_idx):
    # emb holds either the MiniLM matrix or, for tfidf, the period's cached TF-IDF matrix
    # (vocabulary fitted once on the pre-eval train pool), both indexed by eidx
    tr_i, te_i = np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)  # concat can upcast eidx
    return emb[tr_i], emb[te_i]


def make_clf(feat, task, C):
    """Multinomial LR everywhere except TF-IDF x Issue (~80 classes), where lbfgs took >10 min per
    monthly fit on the 16 GB machine; there an SGD logistic model (one-vs-rest) is used instead."""
    if feat == "tfidf" and task == "issue":
        from sklearn.linear_model import SGDClassifier
        return SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=15, tol=None, random_state=0)
    return LogisticRegression(C=C, max_iter=2000)


def fit_score(feat, task, tr, te, r, isets, emb, C):
    y = to_regime(tr["label"].tolist(), tr["regime"].tolist(), r, task, isets)
    keep = np.array([v is not None for v in y])
    tr, y = tr[keep], [v for v in y if v is not None]
    Xtr, Xte = featurize(feat, tr["text"], te["text"], emb, tr["eidx"].values, te["eidx"].values)
    lr = make_clf(feat, task, C).fit(Xtr, y)
    P = lr.predict_proba(Xte)
    k = P.argmax(1)
    return np.array(lr.classes_, dtype=object)[k], P.max(1), len(tr), len(lr.classes_)


class _Indexer:
    """Row access by global eidx into a per-period matrix."""
    def __init__(self, X, eidx):
        self.X, self.pos = X, pd.Series(np.arange(len(eidx)), index=eidx)

    def __getitem__(self, idx):
        return self.X[self.pos.loc[np.asarray(idx)].values]


def job(feat, task, per, m, r, S, isets, emb, C):
    t0 = time.time()
    hist = S[(S["set"] == "train") | ((S["set"] == "audit_trn") & (S["month"] <= mshift(m, -1)))]
    hist = hist[[ORDER.index(x) <= ORDER.index(r) for x in hist["regime"]]]
    ev = S[(S["set"] == "eval") & (S["regime"] == r)]
    cal_months = [mshift(m, -k) for k in range(1, 7)]
    # calibration items of earlier regimes are scored too (labels mapped into r) so that
    # "no reset" policies can be replayed; reset policies use only regime == r
    cal = S[(S["set"] == "audit_cal") & S["month"].isin(cal_months)]
    cal = cal[[ORDER.index(x) <= ORDER.index(r) for x in cal["regime"]]]
    te = pd.concat([ev[ev["month"] == m].assign(role="deploy"), cal.assign(role="cal"),
                    ev[ev["month"] == mshift(m, -1)].assign(role="grid")], ignore_index=True)
    if te.empty:
        return None
    pred, score, ntr, ncls = fit_score(feat, task, hist, te, r, isets, emb, C)
    o = te[["cid", "set", "month", "regime", "label", "role", "rank", "eidx"]].copy()
    o["label_r"] = to_regime(te["label"].tolist(), te["regime"].tolist(), r, task, isets)
    o.loc[o["regime"] == r, "label_r"] = o.loc[o["regime"] == r, "label"]
    o["model_regime"] = r
    o["deploy_month"], o["pred"], o["score"] = m, pred, score
    o["n_train"], o["n_classes"], o["sec"] = ntr, ncls, round(time.time() - t0, 1)
    return o


def static_job(feat, task, per, S, isets, emb, C, start_regime):
    hist = S[S["set"] == "train"]
    hist = hist[hist["regime"] == start_regime]
    te = S[S["set"].isin(["eval", "audit_cal", "audit_trn", "preaudit"])].reset_index(drop=True)
    y = hist["label"].tolist()
    if task == "issue":
        keep = np.array([v in isets[start_regime] for v in y]); hist = hist[keep]; y = [v for v in y if v in isets[start_regime]]
    Xtr, Xte = featurize(feat, hist["text"], te["text"], emb, hist["eidx"].values, te["eidx"].values)
    lr = make_clf(feat, task, C).fit(Xtr, y)
    P = lr.predict_proba(Xte)
    classes = list(lr.classes_)
    pred, score = np.empty(len(te), dtype=object), np.zeros(len(te))
    for r in te["regime"].unique():
        idx = np.where(te["regime"].values == r)[0]
        tgt = LABELS[r] if task == "product" else sorted(isets[r])
        M = np.zeros((len(classes), len(tgt)))
        for i, c in enumerate(classes):
            mm = c if r == start_regime else (map_label(c, start_regime, r) if task == "product" else (c if c in isets[r] else None))
            if mm is not None and mm in tgt:
                M[i, tgt.index(mm)] = 1.0
        Q = P[idx] @ M
        pred[idx], score[idx] = np.array(tgt, dtype=object)[Q.argmax(1)], Q.max(1)
    o = te[["cid", "set", "month", "regime", "label", "rank", "eidx"]].copy()
    o["label_r"], o["model_regime"] = o["label"], o["regime"]
    o["role"], o["deploy_month"], o["pred"], o["score"] = "static", "static", pred, score
    o["n_train"], o["n_classes"], o["sec"] = len(hist), len(classes), np.nan
    return o


def main(data: Path, res: Path, feat: str, task: str, n_jobs: int):
    t0 = time.time()
    out_dir = res / "v1"; out_dir.mkdir(exist_ok=True)
    S_all = pd.read_parquet(data / "samples_v1.parquet")
    S_all["label"] = S_all[task]
    S_all["eidx"] = np.arange(len(S_all))
    emb = None
    if feat == "emb":
        emb = np.load(res / "v1" / "emb_minilm_samples_v1.npy").astype(np.float32)
    isets = issue_sets(data)
    C = 4.0 if feat == "tfidf" else 10.0
    parts = []
    for per in ["A", "D", "B"]:
        S = S_all[S_all["period"] == per]
        if feat == "tfidf":
            # vocabulary fitted once per period on the labeled history available at eval start
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True, dtype=np.float32)
            vec.fit(S.loc[S["set"] == "train", "text"])
            from scipy.sparse import csr_matrix
            Xp = vec.transform(S["text"])
            emb = _Indexer(Xp, S["eidx"].values)
        months = sorted(S.loc[S["set"] == "eval", "month"].unique())
        tasks = [(m, r) for m in months for r in sorted(S.loc[(S["set"] == "eval") & (S["month"] == m), "regime"].unique())]
        res_p = Parallel(n_jobs=n_jobs, verbose=10)(delayed(job)(feat, task, per, m, r, S, isets, emb, C) for m, r in tasks)
        start_regime = S.loc[S["set"] == "train", "regime"].iloc[-1]
        st = static_job(feat, task, per, S, isets, emb, C, start_regime)
        part = pd.concat([x for x in res_p if x is not None] + [st], ignore_index=True)
        part["period"] = per
        parts.append(part)
        print(f"{feat}/{task} period {per}: {len(tasks)} monthly models, t={time.time()-t0:.0f}s", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df["correct"] = df["pred"] == df["label_r"]  # label_r None (no carry-over) -> False; such cal items are dropped by no-reset policies
    df["tier"], df["task"] = ("t0" if feat == "tfidf" else "f"), task
    df.to_parquet(out_dir / f"tier_{feat}_{task}.parquet", index=False)
    print(f"done rows={len(df):,} t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 4)
