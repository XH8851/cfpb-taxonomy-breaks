"""Stream the CFPB narratives archive zips (never extracted) into one parquet of
narrative rows plus monthly label-count tables used to build the label map.

Usage: python -I scripts/01_parse_archive.py <archive_dir> <out_dir>
"""
import sys, glob, zipfile, time
from pathlib import Path
import pandas as pd

KEEP = ["Date received", "Product", "Sub-product", "Issue", "Sub-issue",
        "Consumer complaint narrative", "Company", "State", "Tags",
        "Submitted via", "Complaint ID"]
RENAME = {"Date received": "date", "Product": "product", "Sub-product": "sub_product",
          "Issue": "issue", "Sub-issue": "sub_issue", "Consumer complaint narrative": "text",
          "Company": "company", "State": "state", "Tags": "tags",
          "Submitted via": "via", "Complaint ID": "cid"}


def main(arch: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    narr, counts_all, counts_narr = [], [], []
    t0 = time.time()
    for f in sorted(glob.glob(str(arch / "*.zip"))):
        z = zipfile.ZipFile(f)
        name = z.infolist()[0].filename
        n_rows = n_text = 0
        with z.open(name) as fh:
            for ch in pd.read_csv(fh, usecols=KEEP, dtype=str, chunksize=250_000,
                                  keep_default_na=False):
                ch = ch.rename(columns=RENAME)
                ch["month"] = ch["date"].str[:7]
                n_rows += len(ch)
                counts_all.append(ch.groupby(["month", "product", "issue"]).size()
                                  .rename("n").reset_index())
                t = ch[ch["text"].str.strip() != ""].copy()
                n_text += len(t)
                t["src"] = Path(f).name
                narr.append(t)
                counts_narr.append(t.groupby(["month", "product", "issue"]).size()
                                   .rename("n").reset_index())
        print(f"{Path(f).name}: rows={n_rows:,} narratives={n_text:,} "
              f"t={time.time()-t0:.0f}s", flush=True)
    df = pd.concat(narr, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates("cid")  # same complaint ID appearing in two exports
    df["cid"] = df["cid"].astype("int64")
    df.to_parquet(out / "narratives.parquet", compression="zstd", index=False)
    for nm, parts in [("label_counts_all", counts_all), ("label_counts_narr", counts_narr)]:
        c = pd.concat(parts).groupby(["month", "product", "issue"])["n"].sum().reset_index()
        c.to_parquet(out / f"{nm}.parquet", index=False)
    print(f"narratives total={before:,} unique cid={len(df):,} "
          f"date range {df['date'].min()} .. {df['date'].max()}  t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
