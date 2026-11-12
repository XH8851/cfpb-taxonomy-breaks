"""LLM tier (zero-shot, MLX): route a complaint narrative to one product of the item's regime.

Options are listed with single-letter IDs (A, B, ...) plus a description made of the
regime's sub-product names (the options a consumer sees on the CFPB form). The answer is the
next token after the assistant header, restricted to the option letters; the output head runs
in float32, so the K-way distribution is exact. The shared instruction prefix is cached per
(regime, option order) and trimmed after each item. Narratives are truncated to MAX_TOK tokens.

Usage: python -I scripts/08_llm_tier.py --model <hf id> --items <parquet> --out <parquet>
          [--order forward|reverse|shuffle] [--variant main|para] [--limit N]
          [--fewshot <audit parquet> --k 4]   retrieval few-shot from RELEASED audit labels:
              for an item in month m, candidates are audit items of the same regime from months
              <= m-1 (label delay d=1); top-k by TF-IDF cosine; each example truncated to EX_TOK tokens
items parquet needs: cid, regime, text (product optional, for accuracy printout)
"""
import argparse, json, time, string
from pathlib import Path
import numpy as np
import pandas as pd
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
import sys
sys.path.insert(0, str(Path(__file__).parent))
from labels import LABELS

MAX_TOK = 384
EX_TOK = 100
INSTR = {
    "main": ("You route consumer financial complaints to the product category used by a U.S. "
             "federal regulator's complaint database. Read the complaint and pick the single best "
             "category."),
    "para": ("A consumer wrote the complaint below about a financial product or service. Choose the "
             "one product category, from the list, that this complaint is about."),
}


def descriptions(sub_map: dict, regime: str) -> dict:
    return {p: "; ".join(sorted(sub_map.get(regime, {}).get(p, []))) for p in LABELS[regime]}


class LetterScorer:
    def __init__(self, model, tok, regime, desc, order, variant):
        self.model, self.tok = model, tok
        labs = list(LABELS[regime])
        if order == "reverse":
            labs = labs[::-1]
        elif order == "shuffle":
            labs = [labs[i] for i in np.random.default_rng(17).permutation(len(labs))]
        self.labels = labs
        self.letters = list(string.ascii_uppercase[:len(labs)])
        opts = "\n".join(f"{l}. {p}" + (f" (covers: {desc[p]})" if desc.get(p) else "")
                         for l, p in zip(self.letters, labs))
        self.head = f"{INSTR[variant]}\n\nCategories:\n{opts}\n\nComplaint:\n"
        self.tail = "\n\nAnswer with the letter of the best category only."
        self.ids = []
        for l in self.letters:
            t = tok.encode(l, add_special_tokens=False)
            assert len(t) == 1, (l, t)
            self.ids.append(t[0])
        a, b = self._encode("AAAA one"), self._encode("BBBB two")
        n = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        self.prefix = a[:max(n - 2, 1)]
        self.cache = make_prompt_cache(model)
        self._forward(self.prefix)

    def _encode(self, text):
        msgs = [{"role": "user", "content": self.head + text + self.tail}]
        s = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        assert time.strftime("%Y-%m-%d") not in s
        return self.tok.encode(s, add_special_tokens=False)

    def _forward(self, ids):
        x = mx.array(ids)[None]
        h = self.model.model(x, self.cache)[:, -1:, :].astype(mx.float32)
        m = self.model
        head = m.model.embed_tokens.as_linear if m.args.tie_word_embeddings else m.lm_head
        return head(h)[0, -1]

    def trunc(self, text, n):
        t = self.tok.encode(text, add_special_tokens=False)
        return self.tok.decode(t[:n]) if len(t) > n else text

    def score(self, text, examples=()):
        text = self.trunc(text, MAX_TOK)
        if examples:
            let = dict(zip(self.labels, self.letters))
            shots = "".join(f"[Example] {self.trunc(x, EX_TOK)}\n[Answer] {let[y]}\n\n" for x, y in examples)
            text = "Labeled examples of recent complaints:\n\n" + shots + "[Complaint to route]\n" + text
        ids = self._encode(text)
        p = len(self.prefix)
        assert ids[:p] == self.prefix
        logits = self._forward(ids[p:])
        trim_prompt_cache(self.cache, len(ids) - p)
        sub = np.array(logits)[self.ids].astype(np.float64)
        sub -= sub.max()
        pr = np.exp(sub) / np.exp(sub).sum()
        return pr, len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--submap", required=True, help="json {regime: {product: [sub-products]}}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--order", default="forward")
    ap.add_argument("--variant", default="main")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fewshot", default="")
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()
    df = pd.read_parquet(a.items)
    if a.limit:
        df = df.sample(min(a.limit, len(df)), random_state=0)
    sub_map = json.loads(Path(a.submap).read_text())
    t0 = time.time()
    model, tok = load(a.model)
    load_s = time.time() - t0
    mx.reset_peak_memory()
    rows = []
    pool = pd.read_parquet(a.fewshot) if a.fewshot else None
    for r in sorted(df["regime"].unique()):
        sc = LetterScorer(model, tok, r, descriptions(sub_map, r), a.order, a.variant)
        part = df[df["regime"] == r]
        if pool is not None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            pr_pool = pool[pool["regime"] == r].reset_index(drop=True)
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(pr_pool["text"])
            Xp = vec.transform(pr_pool["text"])
        for i, it in enumerate(part.itertuples()):
            ex = ()
            if pool is not None:
                ok = (pr_pool["month"] < it.month).values
                if ok.any():
                    sims = (Xp @ vec.transform([it.text]).T).toarray().ravel()
                    sims[~ok] = -1
                    top = np.argsort(-sims)[:a.k]
                    ex = [(pr_pool.at[j, "text"], pr_pool.at[j, "product"]) for j in top if sims[j] >= 0][::-1]
            t1 = time.perf_counter()
            pr, ntok = sc.score(it.text, ex)
            sec = time.perf_counter() - t1
            k = int(pr.argmax())
            rows.append({"cid": it.cid, "regime": r, "pred": sc.labels[k], "score": float(pr[k]),
                         "probs": json.dumps({l: float(x) for l, x in zip(sc.labels, pr)}),
                         "n_tok": ntok, "sec": sec, "n_shots": len(ex)})
            if (i + 1) % 200 == 0:
                print(f"{r} {i+1}/{len(part)} {np.mean([x['sec'] for x in rows[-200:]]):.3f}s/item", flush=True)
    out = pd.DataFrame(rows)
    if "product" in df.columns:
        out = out.merge(df[["cid", "product"]], on="cid")
        out["correct"] = out["pred"] == out["product"]
        print("accuracy", round(out["correct"].mean(), 4))
    out.to_parquet(a.out, index=False)
    meta = {"model": a.model, "order": a.order, "variant": a.variant, "fewshot": bool(a.fewshot), "k": a.k, "n": len(out),
            "load_s": load_s, "peak_mem_gb": mx.get_peak_memory() / 1e9,
            "sec_p50": float(out["sec"].median()), "sec_p95": float(out["sec"].quantile(0.95)),
            "tok_mean": float(out["n_tok"].mean()), "score_distinct": int(out["score"].nunique())}
    Path(a.out + ".meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
