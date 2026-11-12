# Risk-Controlled Complaint Routing Across Real Taxonomy Breaks: benchmark code and data

Code and data for the paper *Risk-Controlled Complaint Routing Across Real Taxonomy Breaks* (under review). This repository lets a reader:
- replay every result in the paper **without a GPU**, from the cached per-item tier outputs;
- rebuild the full pipeline from the public CFPB narratives archive.

## Layout
The layout mirrors the project, so every script runs unchanged from the repository root.
- `results/v1/`: cached tier outputs and every result file used by the paper.
  - `tier_*.parquet`: long tables keyed by `(period, deploy_month, model_regime, role, cid)` with `score`, `pred`, `correct`. Roles are `deploy` (month-m stream), `cal` (released calibration labels of months m−6..m−1), `grid` (month m−1 stream, used unlabeled) and `static` (eval-start model).
  - `main_monthly.csv` / `main_summary.csv`: per-policy monthly and summary metrics (A, B; Product and Issue).
  - `dev_*`: period D, including the knob grid and the frozen knobs (`frozen_knobs*.json`).
  - `contrasts_alpha*.csv`: headline contrasts C1–C4 with block-bootstrap CIs and Holm p-values.
  - `secondary.json`, `analysis.json`, `cost_table.csv`, `c4_drop_rule.json`, `resample_monthly.csv`, `contam_scores.parquet`, `llm_scores_*.parquet`.
- `data/`
  - `samples_v1.parquet`: every sample row with complaint ID, period, role, month, regime, product, issue, stream rank and near-duplicate group. **No narrative text.**
  - `label_counts_narr.parquet`: monthly label counts of the full archive.
  - `submap_v1.json`, `SHA256SUMS.txt` (the 21 archive files), `label_dates.md`.
- `audit/noise_audit_300.csv`: the 300-item label-noise audit (complaint IDs, regime, label, judgment). It was annotated by an LLM under the rubric in the paper's appendix, without human adjudication.
- `PROTOCOL.md`: the protocol fixed before the final replays, with dated amendments and departures.
- `DATASHEET.md`, `requirements.txt` (Python 3.12).
- `code/`: all scripts. `labels.py` holds the regimes and the typed crosswalk. `16_harness.py` is the replay harness.

## Replay the paper's numbers (CPU, a few minutes, no GPU, no text needed)
```bash
git clone https://github.com/XH8851/cfpb-taxonomy-breaks.git && cd cfpb-taxonomy-breaks
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp -r results/v1 results/v1_shipped                                 # keep the shipped copy
.venv/bin/python -I code/18_replay_v1.py data results --no-llm        # all policies, Product + Issue
.venv/bin/python -I code/18_replay_v1.py data results --only-llm      # adds the LLM contrast C4
.venv/bin/python -I code/18b_replay_issue.py data results             # Issue with its own frozen knobs
.venv/bin/python -I code/27_compare_replay.py results                 # compare with the shipped copy
for a in 0.10 0.15 0.05; do .venv/bin/python -I code/19_contrasts.py results $a; done
```
The weighted conformal-risk-control baseline (`wcrc`) needs the MiniLM embeddings (315 MB, not shipped). Without them the replay skips that one policy and says so. To regenerate them, rebuild the narrative text (below) and run `code/15_embed.py` (about 30 min on Apple MPS).

## Evaluate a new acceptance rule
A rule is a function `choose(cal, grid, tiers, alpha, ...) -> (config, thresholds)` in `16_harness.py`; `run(...)` applies it month by month under the shared label economy (W, delay, regime reset) and returns monthly errors, automation and new-class outcomes; `20_analysis.py` adds audit and review costs.

## Rebuild from the public archive
1. Download the 21 zips from the CFPB FOIA reading room (URL in `DATASHEET.md`) and check `SHA256SUMS.txt`.
2. Run `01_parse_archive.py` → `02v1_dedup.py` → `03v1_samples.py` → `14_monthly_tiers.py` / `15_embed.py` → `06v1_modernbert.py` → `08v1_llm_prep.py` + `08_llm_tier.py` → `17_freeze_dev.py` → `18_replay_v1.py` / `18b_replay_issue.py` → `19_contrasts.py` → `20_analysis.py` → `21_make_tables.py`.

`run_phase3_*.sh` show the exact order used. Total about 35 machine-hours for the final runs on a MacBook Air M4 (16 GB), plus about 8 h for the pilot.

## Licenses
- Code (`code/`): MIT, see `LICENSE`.
- Derived benchmark files (`data/`, `results/`, `audit/`, documentation): CC BY 4.0, see `LICENSE-DATA`.
- Source data: the CFPB Consumer Complaint Database narratives are public-domain (CC0) records published by the Consumer Financial Protection Bureau. This repository contains no narrative text; `code/01_parse_archive.py` rebuilds it from the public archive.

## Citation
See `CITATION.cff` (GitHub shows a "Cite this repository" button). The archived release has a Zenodo DOI.

## Privacy
The narratives were scrubbed by the CFPB (masked as XXXX). Do not attempt to re-identify consumers or companies' employees.
