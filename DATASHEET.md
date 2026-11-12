# Datasheet: CFPB Taxonomy-Break Routing Benchmark

Follows the questions of Gebru et al., *Datasheets for Datasets* (CACM 2021).

## Motivation
- **Purpose.** Measure whether error guarantees for automatically routed consumer complaints survive real, dated changes of the label taxonomy, and at what automation and human-review cost.
- **Gap.** Existing evaluations of certified or cascaded prediction use simulated shifts without real timestamps, label-set change and a human tier at the same time.
- **Creator and funding.** One independent researcher; no external funding.

## Composition
- **Instances.** Consumer complaint narratives from the CFPB Consumer Complaint Database narratives archive. Each instance is identified by its CFPB `Complaint ID` (`cid`) and carries the received date, product, sub-product and issue.
- **Size.** 429,700 sample rows over 398,500 distinct complaints, in three periods:
  - **A:** 146,700 rows; evaluation months 2016-04..2018-12; product break 2017-04-24.
  - **D (development):** 141,500 rows; 2019-01..2021-07; no product break; placebo dates 2019-10-01 and 2020-10-01.
  - **B:** 141,500 rows; 2022-08..2025-02; product break 2023-08-25.
- **Roles per period.**
  - `train`: up to 5,000 per month in the 12 months before evaluation.
  - `preaudit`: 300 per month in the last 3 pre-evaluation months.
  - `audit_cal` and `audit_trn`: 300 each per evaluation month.
  - `eval`: 2,000 per evaluation month; `rank` 0..1,999 is a fixed random order, and LLM configurations use rank < 1,000.
- **Labels.** The consumer-selected product and issue, interpreted in the taxonomy regime in force on the received date (R0, R1, R2).
  - The typed crosswalk (same / rename / merge / successor / none) is in `code/labels.py`.
  - New classes at a break: *Vehicle loan or lease* (2017); *Prepaid card* and *Debt or credit management* (2023).
- **Model outputs.** Cached per-item scores of three tiers, keyed by (period, deploy month, model regime, role, cid):
  - TF-IDF + LR, retrained monthly
  - ModernBERT-base, 4 models
  - Qwen3-4B 4-bit with retrieval few-shot prompting, months around each break

  Plus a frozen MiniLM embedding + LR tier (`outputs/tier_*.parquet`).
- **Text.** The benchmark release contains complaint IDs, not narrative text. A script rebuilds the text from the public archive (sha256 of the 21 archive files are in `data/SHA256SUMS.txt`).
- **Label noise.** In a 300-item audit, 4.7% of product labels were judged clearly wrong and 8.7% ambiguous. The audit was annotated by an LLM under a written rubric, without human adjudication (`audit/noise_audit_300.csv`).
- **Personal information.** Narratives are scrubbed by the CFPB (XXXX masking) and published with consumer consent. No attempt was made to re-identify or link records.

## Collection process
- **Source.** 21 export zip files from https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/, downloaded 2026-10-31 (CC0, public record).
- **Frozen archive.** The CFPB stopped publishing narratives on 2026-08-14, so the archive is frozen.
- **Sampling.** Uniform within month after one fixed shuffle (seed 20261101). Near-duplicate groups come from MinHash LSH (128 permutations, word 5-grams, Jaccard ≥ 0.8) on a pool capped at 20,000 complaints per month.

## Preprocessing
- Rows without narratives are dropped.
- Break dates are read from the first and last day each product label appears.
- 5 complaints whose label does not belong to their date's regime are dropped.
- Issue classes need at least 50 narratives in a regime.

## Uses
- **Intended.** Benchmarking acceptance and abstention rules, cascades, recalibration and drift detection under real label-space change; cost and automation trade-offs with a human tier.
- **Not intended.** Inferring facts about individual consumers or companies; evaluating companies' complaint handling; any re-identification.
- **Known limits.**
  - One regulator and two breaks.
  - Consumer-selected labels are noisy.
  - Post-2023 narratives contain LLM-assisted text.
  - Volume grows 6× from window A to B.

## Distribution
- **Distribution.** GitHub repository (https://github.com/XH8851/cfpb-taxonomy-breaks), archived on Zenodo with a DOI for each release.
- **Licenses (proposed; author to confirm).** Code under MIT; derived benchmark files under CC BY 4.0. The underlying CFPB data are CC0.

## Maintenance
- Maintained by the author.
- The source archive is frozen, so the benchmark files will not change. Errata will be versioned on Zenodo.
