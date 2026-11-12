"""Build the submission supplement (code + cached per-item outputs + ID lists + docs), with no
local paths or personal identifiers. Usage: python -I scripts/23_build_supplement.py <project_root>
Output: supplement/T2_supplement.zip (+ supplement/build/ tree)
"""
import sys, re, shutil, zipfile, hashlib
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1]); out = root / "supplement" / "build"
if out.exists():
    shutil.rmtree(out)
(out / "code").mkdir(parents=True); (out / "results" / "v1").mkdir(parents=True); (out / "results" / "pilot").mkdir(); (out / "data").mkdir()
ROOT_STR = str(root); RAW_STR = str(root.parent / "shared_data")
SUB = [(ROOT_STR, "$ROOT"), (RAW_STR, "$ROOT/data/raw"), (str(Path.home()), "$HOME")]

CODE = ["01_parse_archive.py", "02_dedup_windows.py", "02v1_dedup.py", "03v1_samples.py", "03bv1_submap.py", "06v1_modernbert.py",
        "08_llm_tier.py", "08v1_llm_prep.py", "08c_contam_items.py", "13_llm_configs.py", "14_monthly_tiers.py", "15_embed.py",
        "16_harness.py", "17_freeze_dev.py", "17b_freeze_dev_issue.py", "18_replay_v1.py", "18b_replay_issue.py", "19_contrasts.py", "19b_c4_drop_rule.py", "24_crosswalk_table.py", "23_build_supplement.py", "25_plain_abstract.py", "26_secondary.py", "27_compare_replay.py",
        "20_analysis.py", "21_make_tables.py", "22_bench_stats.py", "labels.py", "manifest.py", "make_bib.py", "check_citations.py",
        "run_phase3_cpu1.sh", "run_phase3_freeze.sh", "run_phase3_gpu.sh"]
for f in CODE:
    t = (root / "scripts" / f).read_text()
    for a, b in SUB:
        t = t.replace(a, b)
    t = re.sub(r"P=\$ROOT\b", 'P="${ROOT:-$(cd "$(dirname "$0")/.." \\&\\& pwd)}"', t) if f.endswith(".sh") else t
    (out / "code" / f).write_text(t)
OUTS = ["tier_tfidf_product.parquet", "tier_emb_product.parquet", "tier_tfidf_issue.parquet", "tier_emb_issue.parquet",
        "tier_mbert_product_A.parquet", "tier_mbert_product_B.parquet", "tier_llm_product_A.parquet", "tier_llm_product_B.parquet",
        "llm_scores_A.parquet", "llm_scores_B.parquet", "contam_scores.parquet", "main_monthly.csv", "main_summary.csv",
        "resample_monthly.csv", "dev_factorial_summary.csv", "dev_factorial_monthly.csv", "dev_factorial_issue_summary.csv",
        "dev_knob_grid.csv", "dev_knob_grid_issue.csv", "frozen_knobs.json", "frozen_knobs_issue.json", "contrasts_alpha0.05.csv",
        "contrasts_alpha0.10.csv", "contrasts_alpha0.15.csv", "c4_drop_rule.json", "analysis.json", "cost_table.csv",
        "mbert_meta_A.json", "mbert_meta_B.json"]
# results mirror the project layout so every script runs unchanged from the supplement root
for f in OUTS + ["secondary.json"]:
    shutil.copy(root / "results" / "v1" / f, out / "results" / "v1" / f)
shutil.copy(root / "results" / "pilot" / "llm_configs.json", out / "results" / "pilot" / "llm_configs.json")
shutil.copy(root / "results" / "dedup_stats_v1.json", out / "results" / "dedup_stats_v1.json")
# ID lists (no text): rebuild samples from the public archive
# samples WITHOUT narrative text (labels are CFPB CC0 metadata); same file name the scripts read
S = pd.read_parquet(root / "data" / "samples_v1.parquet", columns=["cid", "period", "set", "month", "date", "regime", "product", "issue", "rank", "group", "group_size"])
S.to_parquet(out / "data" / "samples_v1.parquet", index=False)
shutil.copy(root / "data" / "label_counts_narr.parquet", out / "data" / "label_counts_narr.parquet")
shutil.copy(root / "data" / "submap_v1.json", out / "data" / "submap_v1.json")
shutil.copy(root.parent / "shared_data" / "cfpb_narratives" / "SHA256SUMS.txt", out / "data" / "SHA256SUMS.txt")
shutil.copy(root / "results" / "label_dates.md", out / "data" / "label_dates.md")
shutil.copy(root / "env_pip_freeze.txt", out / "requirements.txt")
shutil.copy(root / "supplement" / "DATASHEET.md", out / "DATASHEET.md")
shutil.copy(root / "supplement" / "README.md", out / "README.md")
# label-noise audit (no narrative text) and the dated specification (paths sanitized)
(out / "audit").mkdir(exist_ok=True)
shutil.copy(root / "review" / "noise_audit" / "annotations_agent.csv", out / "audit" / "noise_audit_300.csv")
shutil.copy(root / "supplement" / "PROTOCOL.md", out / "PROTOCOL.md")  # neutral dated protocol (internal SPEC/LEDGER are not shipped)
# privacy / path scan
import os
_terms = [re.escape(str(Path.home())), re.escape(Path.home().name), re.escape(os.sep + "Users" + os.sep)] + [re.escape(x) for x in os.environ.get("PRIVACY_TERMS", "").split(",") if x]
bad = re.compile("|".join(_terms))  # identifiers are taken from the machine/env at build time, never stored in code
hits = []
for p in out.rglob("*"):
    if p.is_file() and p.suffix in {".py", ".sh", ".md", ".json", ".csv", ".txt"}:
        for i, line in enumerate(p.read_text(errors="ignore").splitlines()):
            if bad.search(line):
                hits.append(f"{p.relative_to(out)}:{i+1}: {line[:120]}")
assert not hits, "\n".join(hits)
z = root / "supplement" / "T2_supplement.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(out.rglob("*")):
        if p.is_file():
            zf.write(p, Path("T2_supplement") / p.relative_to(out))
print(f"files={sum(1 for p in out.rglob('*') if p.is_file())} zip={z.stat().st_size/1e6:.1f} MB sha256={hashlib.sha256(z.read_bytes()).hexdigest()[:16]} path/privacy hits=0")
