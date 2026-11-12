"""Build the two study windows, near-duplicate groups (MinHash LSH, Jaccard>=0.8)
and group-disjoint roles (train / audit / eval).

Usage: python -I scripts/02_dedup_windows.py <data_dir>
Inputs : data/narratives.parquet
Outputs: data/window_items.parquet, results/dedup_stats.json
"""
import sys, re, json, hashlib, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH

NPERM, THRESH, SHINGLE = 128, 0.8, 5
CAP_PER_MONTH = 20_000
SEED = 20261031
WINDOWS = {"A": ("2015-03", "2018-12"), "B": ("2021-08", "2025-02")}
_TOK = re.compile(r"[a-z]+")


def norm_tokens(t: str) -> list[str]:
    t = t.lower().replace("xxxx", " ").replace("xx/xx/", " ")
    return _TOK.findall(t)


def minhash_vals(text: str) -> np.ndarray:
    w = norm_tokens(text)
    sh = {" ".join(w[i:i + SHINGLE]) for i in range(max(1, len(w) - SHINGLE + 1))} or {""}
    m = MinHash(num_perm=NPERM, seed=1)
    m.update_batch([s.encode() for s in sh])
    return m.hashvalues


def role_of(key: str, salt: str) -> str:
    h = int(hashlib.sha256((salt + key).encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 60 else ("audit" if h < 75 else "eval")


def main(data: Path) -> None:
    t0 = time.time()
    df = pd.read_parquet(data / "narratives.parquet",
                         columns=["cid", "date", "month", "product", "sub_product", "issue",
                                  "text", "company", "via", "tags"])
    parts = []
    for w, (a, b) in WINDOWS.items():
        s = df[(df["month"] >= a) & (df["month"] <= b)].copy()
        s["window"] = w
        parts.append(s)
    s = pd.concat(parts, ignore_index=True)
    n_window_all = s.groupby("window").size().to_dict()
    # cap per month (uniform sample) before dedup; dedup only matters among items we use
    s = (s.groupby("month", group_keys=False)
           .apply(lambda g: g.sample(min(len(g), CAP_PER_MONTH), random_state=SEED)))
    s = s.reset_index(drop=True)
    print(f"pool {len(s):,} items ({n_window_all}) t={time.time()-t0:.0f}s", flush=True)

    with Pool(8) as p:
        hv = p.map(minhash_vals, s["text"].tolist(), chunksize=2000)
    print(f"minhash done t={time.time()-t0:.0f}s", flush=True)

    lsh = MinHashLSH(threshold=THRESH, num_perm=NPERM)
    parent = np.arange(len(s))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Only cluster representatives are inserted into the index: an item that matches any
    # representative joins (and merges) those clusters and is not inserted itself. This keeps
    # LSH buckets small despite very large template clusters (quadratic otherwise).
    reps = {}
    for i, v in enumerate(hv):
        m = MinHash(num_perm=NPERM, seed=1, hashvalues=v, scheme="affine32")
        hit = False
        for j in lsh.query(m):
            if m.jaccard(reps[j]) >= THRESH:
                hit = True
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
        if not hit:
            lsh.insert(i, m)
            reps[i] = m
        if i % 200_000 == 0:
            print(f"lsh {i:,} t={time.time()-t0:.0f}s", flush=True)
    roots = np.array([find(i) for i in range(len(s))])
    s["group"] = s["cid"].values[roots]          # group id = cid of the root member
    s["group_size"] = s.groupby("group")["cid"].transform("size")
    s["role_group"] = [role_of(str(g), "grp") for g in s["group"]]
    s["role_random"] = [role_of(str(c), "rnd") for c in s["cid"]]
    s.to_parquet(data / "window_items.parquet", compression="zstd", index=False)

    dup = (s["group_size"] > 1)
    stats = {
        "pool_items": int(len(s)), "window_items_before_cap": n_window_all,
        "share_in_dup_groups": float(dup.mean()),
        "share_in_dup_groups_by_window": s.assign(d=dup).groupby("window")["d"].mean().to_dict(),
        "n_groups": int(s["group"].nunique()),
        "largest_groups": s.groupby("group").size().sort_values(ascending=False).head(10).tolist(),
        "role_counts_group": s["role_group"].value_counts().to_dict(),
        "params": {"num_perm": NPERM, "threshold": THRESH, "shingle_words": SHINGLE,
                   "cap_per_month": CAP_PER_MONTH, "seed": SEED},
        "seconds": round(time.time() - t0),
    }
    out = data.parent / "results" / "dedup_stats.json"
    out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
