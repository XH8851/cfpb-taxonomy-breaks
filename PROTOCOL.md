# Evaluation Protocol (pre-specified 2026-10-31; amendments dated below)

This document records the protocol of the paper *Risk-Controlled Complaint Routing Across Real Taxonomy Breaks*. It was fixed before the final replays on windows A and B. It was not registered with a third party; the dates are those of the project's own records.

## 1. Data and periods
- **Source.** CFPB Consumer Complaint Database narratives archive: 21 files, CC0, 3,851,415 narratives, 2015-03-19..2026-08-11.
- **Taxonomy regimes.** Product label sets R0 (until 2017-04-22), R1 (from 2017-04-24) and R2 (from 2023-08-25), with a typed crosswalk (`code/labels.py`).
- **Periods.**

  | Period | Labeled history | Evaluation months | Break / placebo |
  |---|---|---|---|
  | A | 2015-04..2016-03 | 2016-04..2018-12 | break 2017-04-24 |
  | D (development) | 2018-01..2018-12 | 2019-01..2021-07 | placebo dates 2019-10-01, 2020-10-01 |
  | B | 2021-08..2022-07 | 2022-08..2025-02 | break 2023-08-25 |

- **Label economy.**
  - Labeled history ends at evaluation start.
  - Each evaluation month has a stream of 2,000 complaints and an audit of 600: 300 calibration labels and 300 training labels, all charged as human review.
  - Labels of month m become usable in month m+1.
  - Calibration labels are never used for training.

## 2. Tiers
- **T0:** TF-IDF + logistic regression, retrained monthly.
- **T1:** ModernBERT-base; 4 fine-tunes of 30,000 complaints (at evaluation start and three months after the break, per window).
- **T2:** Qwen3-4B 4-bit, scored for break−2..break+6 on the first 1,000 stream complaints.
- **F:** frozen MiniLM + logistic regression, used for period D and the issue task.
- **Human tier:** takes all deferred complaints.

## 3. Acceptance rules
- **Certify-or-defer (CoD).** Thresholds come from quantiles of the previous month's unlabeled scores along fixed-sequence paths, one per non-empty subset of tiers. Each path is tested with a one-sided Clopper–Pearson bound at δ/P on released calibration labels of the last W months; δ = 0.05.
- **Factorial.** Threshold update (static, fully static, rolling) × rule (plug-in, CoD) × reset (known date, none, χ² detector).
- **Further baselines.** Rolling conformal risk control, weighted + recency conformal risk control, adaptive conformal inference, and B1 (static encoder threshold).

## 4. Knob selection (on period D only)
- **Candidates:** W ∈ {3, 6}, ℓ0 ∈ {0.10, 0.30}, test ∈ {single path, multi-path}.
- **Rule:** maximize automation at α = 0.10 subject to at most 1 excess error per 1,000 overall and within ±3 months of each placebo.
- **Selected:**
  - product routing: W = 6, ℓ0 = 0.30, multi-path;
  - issue routing: W = 6, ℓ0 = 0.10, multi-path.
- **ACI step size:** none met the rule; the least-excess candidate (0.005) was used.

## 5. Metrics and contrasts
- **Primary metric.** Excess errors per 1,000 complaints, clipped per month, reported with automation. Also reported: months whose 95% Clopper–Pearson lower bound exceeds α (non-vacuous months only), and vacuous months (< 50 automated complaints).
- **Levels.** α ∈ {0.10, 0.15} main; 0.05 stress.
- **Headline contrasts** (Holm-corrected, 3-month block bootstrap):
  - C1: rolling CoD vs static plug-in, all months.
  - C2: rolling CoD vs rolling plug-in, all months.
  - C3: reset vs no reset, break..break+6.
  - C4: cascade with vs without the LLM tier, break..break+6.
- **LLM drop rule.** If the LLM tier adds < 2 points of certified automation in at least half of the months break..break+6 at α = 0.10, it is dropped from the recommended system.

## 6. Amendments and departures (all disclosed in the paper's appendix)

1. **2026-10-31.** The LLM prompting (retrieval few-shot, k = 4) was chosen after a pilot on 1,883 post-break complaints from windows A and B. It exceeded the pre-set cost cap (2× single-order cost).
2. **2026-10-31.** The multi-path test was added after a synthetic test showed that a single path lets one weak tier block certification. This was before any real replay.
3. **2026-11-02.** The issue-routing knobs were re-selected on period D after the first replay showed that the product knobs certified nothing for issues. This was the one corrective re-run of the issue experiments.
4. **2026-11-02.** A counting error (pre-break *Vehicle loan or lease* counted as new in window B) was fixed and the replays re-run. Errors and automation were unchanged.
5. **2026-11-02.** C4 was first computed on break−2..break+6 (p = 0.13) and then restored to the pre-specified window break..break+6.
6. **Planned but not done:**
   - a human check of the label-noise audit (the audit is LLM-annotated without human adjudication);
   - a measured human error rate;
   - an adjudicated-label sensitivity analysis;
   - a d = 2 label-delay ablation;
   - a baseline-fidelity table;
   - ACI native-target results;
   - p95 latency;
   - an LLM API cost.

   The released monthly files contain the inputs for the analyses that do not need new labels.
7. **2026-11-04.** A secondary comparison chooses the plug-in and CRC nominal levels on period D with the knob-selection rule (`code/26_secondary.py`). It is reported as secondary to C1–C4.
