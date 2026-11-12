"""Frozen all-MiniLM-L6-v2 embeddings (mean-pooled, normalized, max 256 tokens) for every row of
data/samples_v1.parquet, in row order (row i -> eidx i). Output: results/v1/emb_minilm_samples_v1.npy (float16).
Usage: python -I scripts/15_embed.py <data_dir> <results_dir>"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

data, res = Path(sys.argv[1]), Path(sys.argv[2])
S = pd.read_parquet(data / "samples_v1.parquet", columns=["cid", "text"])
# embed each distinct cid once (the same item can appear in two periods), then expand
u = S.drop_duplicates("cid").reset_index(drop=True)
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="mps")
enc.max_seq_length = 256
t0 = time.time()
E = enc.encode(u["text"].tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=False,
               convert_to_numpy=True).astype(np.float16)
pos = pd.Series(np.arange(len(u)), index=u["cid"])
np.save(res / "v1" / "emb_minilm_samples_v1.npy", E[pos.loc[S["cid"]].values])
print(f"rows={len(S):,} unique={len(u):,} t={time.time()-t0:.0f}s")
