# GLP-1 clinical-impact analysis code — Part II v2

This is the analysis code for the GLP-1 receptor agonist weight and HbA1c study: the
statistical pipeline that produces the results reported in the manuscript and supplementary
materials. It is the set of step scripts re-run across each persistence-of-therapy definition
(30–730 days) by the batch runners.

**This is version 2, dated 2026-08-17.** It is a corrected copy of the package submitted on
2026-07-22, revised in response to the code review of "Code Part II: GLP-1 Analysis". It adds
no analyses and changes no reported number. `CHANGE_LIST.md` gives the per-file account of
what changed, which review item each change answers, and whether any output changed;
`VERIFICATION.md` records the checks that were run and their results.

Scope note: this package is the analysis code only. Downstream deliverable builders — the
xlsx/PDF table formatters, the multi-panel supplementary figure assemblers, and the
patient-flow figure script — are not included. The extraction/validation study (physician
adjudication, benchmark validation, knowledge-graph construction) is supplied separately, so
Table 1, Extended Data Table 1, Extended Data Table 5 and Extended Data Figs. 1–4 are not produced here.

## Environment
Python 3.13.5. `pip install -r requirements.txt`.

Scripts anchor paths to the study root via `Path(__file__).resolve().parents[...]` and are run
**from the study root**, e.g. `python3 code/step1_prepare_analysis_dataset.py`.

**The source data are not distributed with this code and cannot be redistributed** (see Data availability below), so the pipeline cannot be re-run end to end outside RespondHealth. The code is published so that every analytic decision behind the paper can be read, checked and reused. `code/tests/` runs standalone and needs no study data.

## Contents at a glance

| File | Role |
|---|---|
| `persistence.py` | **The** persistence-of-therapy rule — one implementation, shared by step1 and step0a |
| `analysis_config.py` | Confidence level → normal critical value; replaces 61 hard-coded `1.96`s |
| `model_spec.py` | Spline degrees of freedom and the ordered HbA1c categorical — never silently defaulted |
| `covariates.py` | Covariate selection with every drop logged |
| `gap_grids.sh` | Single source of truth for the persistence grids, sourced by all four runners |
| `tests/` | Regression tests for each correction |
| `step*.py`, `conf_int/`, `structured_only/`, `add_unstructured/` | The pipeline itself |

**On the step numbering.** The top-level files run step0, step0a, step1, step2, step3, then
step5, step6b, step6c, step6d, step8, step8b. The gap is not missing code: **step4** (and
**step6**/**step6cc**) live under `conf_int/gap_120/`, the confidence-interval scripts run at the
prespecified 120-day persistence definition. There is **no step7 script in this package** — step7
produced auxiliary adherence-count tables that are not part of the reported analysis; `step5_*`
reads `step7_adherence_counts/adherence_counts.csv` only if it happens to be present, and
annotates without it otherwise.

---

## 1. Baseline window and day-0 anchoring

This section states the design in the same terms as the Methods, then states plainly what the
code does with it, so that a reader of the code alone reaches the same understanding as a
reader of the paper.

### What the Methods say

Under *Data cleaning, variable derivation and cohort construction*, subsection *Baseline
definition and baseline window*:

> Baseline GLP-1 RA exposure was defined as the first qualifying GLP-1 RA medication event. A
> baseline window spanning −60 to +14 days around this date was used to derive baseline
> measurements. Within this window, the value closest in time to the baseline date was
> selected for each variable (HbA1c, weight, BMI, height), with ties resolved in favor of
> values occurring on or before baseline.

And under *Longitudinal cohort definition and persistence of therapy*:

> To ensure consistent temporal alignment across patients, all individuals contributed a
> baseline observation at day 0 corresponding to treatment initiation.

The window is asymmetric (−60/+14). An earlier draft of the Methods described it as
"symmetric"; that word has been removed as part of this response.

### What the code does — weight

`_ensure_baseline_rows` in `step1_prepare_analysis_dataset.py` implements the day-0 anchoring.

Take a patient whose baseline weight was measured 23 days before starting treatment. That
patient has a valid baseline value under the window rule, but no data row on day 0 itself.
The function creates that day-0 row, carrying the patient's real recorded
`baseline_weight_final`. The outcome on the row, `pct_weight_change`, is `0.0` — not an
imputed value but the definitional one, because it is the baseline compared with itself. The
row writes down the starting point of the trajectory the model estimates.

**At the primary 120-day persistence definition this affects 2,967 of the 16,061 patients
(18.5%)**, spread evenly across the four glycemic groups (17.4–19.6%).

Every such row carries **`baseline_carried_to_day0 = 1`**; observed rows carry `0`. The column is listed
in `keep_cols` and reaches every `analysis_ready_gap*.csv`, so anchored and measured rows can
be told apart downstream and a leave-them-out sensitivity analysis needs no re-derivation of
which rows these were. (That analysis was run for this response: removing all 2,967 rows
changes the 6- and 12-month weight-change estimates by at most 0.024 percentage points.)

The created row also sets `glp1_event_for_adherance = 1`, recording treatment initiation as
the first tick of the persistence clock — a documented event every patient in the cohort has
by construction of the cohort. For 973 patients (6.1%) that day-0 record was the only
medication evidence *in the weight dataset*; all 973 have genuine documented GLP-1 evidence in
the source record, on clinic-visit rows that carry no weight measurement and are therefore not
kept by the weight dataset.

### What the code does — HbA1c: nothing is anchored

`_ensure_baseline_rows_a1c` in `step1_prepare_analysis_dataset_a1c.py` is structurally the twin
of the weight version, but **on this data it creates no rows, and no HbA1c value is anchored to
day 0.**

An HbA1c value appears in this dataset only on the date it was measured; this was confirmed
across the full source. Baseline HbA1c is the measurement closest to initiation within the
−60/+14-day window. For 38.0% of the cohort that measurement fell on the index date itself, so
those patients have a genuine day-0 row; for the remaining 62.0% it fell earlier or later in
the window, and **those patients contribute no observation at day 0 rather than an anchored
one.** The HbA1c models therefore contain no created rows and no zero-variance points at
t = 0.

`baseline_carried_to_day0` is nonetheless written to the HbA1c outputs, where **it sums to zero**. The
column is emitted even though it is everywhere `0` so that a reader can confirm the absence
from the data rather than take it on trust.

---

## 2. Script-to-deliverable map

**The manuscript numbers were produced by `rerun_conf_int_clean_full.sh`**, the end-to-end
runner, reading
`root_data/merged/step8g_with_unstructured_flags_with_assessments_weightcleaned.csv`. The other
three runners re-run subsets of the same step scripts; see §4 for which persistence grid each
one uses.

### Data preparation

| Script | Produces | Manuscript / supplementary item |
|---|---|---|
| `step1_prepare_analysis_dataset.py` | `analysis_ready_gap{g}.csv` (weight) | Input to every weight result |
| `step1_prepare_analysis_dataset_a1c.py` | `analysis_ready_a1c_gap{g}.csv` | Input to every HbA1c result |
| `persistence.py` | the persistence rule used by the above | Supplementary Method S3, *Operationalization of GLP-1 RA persistence of therapy* |
| `structured_only/step0_prefilter_raw.py` | `prefiltered_structured_only.csv` | Structured-only sensitivity cohort |
| `structured_only/gap120/step0_filter_to_structured.py` | structured-only gap-120 subset | Structured-only sensitivity cohort |
| `add_unstructured/prepare_assessment_data.py` | assessment-domain base data | Input to the exploratory note-derived results |

### Cohort description and attrition

| Script | Produces | Manuscript / supplementary item |
|---|---|---|
| `step0_analysis_population_table.py` | baseline characteristics by glycemic category | Main-text cohort characteristics table |
| `step0a_samplesize_analysis.py` | `samplesize_by_month.csv` | **Supplementary Table 1**, *Cohort attrition across alternative GLP-1 persistence definitions* |
| `add_unstructured/freq/population_description.py` | note-derived assessment population description | **Supplementary Table 2** |
| `add_unstructured/freq/mention_frequency.py` | pre/post mention counts by domain | Supporting counts for the note-derived results |

### Model selection and primary trajectory models

| Script | Produces | Manuscript / supplementary item |
|---|---|---|
| `step2_select_spline_df.py`, `step2_select_spline_df_a1c.py` | `model_config.json` / `model_config_a1c.json` (`best_df`) | Spline df selection (QIC/QICu); df = 3 |
| `step3_fit_gee_baseline.py`, `step3_fit_gee_baseline_a1c.py` | `coefficients.csv`, `model_summary.txt` | The reported GEE trajectory models. Run as `_nomet` (metformin excluded — a reporting choice, not a data change) |
| `conf_int/gap_120/step4_predictive_plots.py`, `..._a1c.py` | model-predicted trajectories with CIs | Main-text trajectory figures; **Extended Data Table 3** |
| `conf_int/gap_120/step4_observed_summary_plots.py` | observed binned summaries | Observed-data companions to the trajectory figures |

### Stratified trajectories and contrasts

| Script | Produces | Manuscript / supplementary item |
|---|---|---|
| `step5_forest_contrasts_weight.py`, `step5_forest_contrasts_a1c.py` | `gee_combined_coefficients.csv`, forest plots, contrasts vs Normal Glycemia | Primary stratified comparison across baseline glycemic categories (**Table 3**) |
| `conf_int/gap_120/step6_stratified_by_covariates_weight.py`, `..._a1c.py` | trajectories within strata of age, sex, race, baseline BMI | **Extended Data Table 3**; supplementary stratified figure sets (not numbered display items in the final paper) |
| `step6b_stratified_contrasts_weight.py`, `..._a1c.py` | stratified contrasts at day 365 | Secondary descriptive contrasts |
| `step6c_stratified_forest_plots_weight.py`, `step6c_stratified_forest_plots.py` | stratified forest plots | Supplementary stratified figure sets |
| `conf_int/gap_120/step6cc_3waystrat_covariates_weight.py`, `..._a1c.py` | three-way strata (glycemic × age × sex) | Supplementary three-way stratified figures |
| `step6d_groups_by_glp1.py` | semaglutide-only vs tirzepatide-only comparison | Agent-level comparison |

### Time-to-event

| Script | Produces | Manuscript / supplementary item |
|---|---|---|
| `step8_survival_time_to_weight_loss.py` | `step8_weight_time_to_threshold_events.csv` (5/10/15%) | Event tables for the weight survival analyses |
| `step8_survival_time_to_a1c_drop.py` | `step8_a1c_time_to_threshold_events.csv` (0.5/1.0/1.5 pt) | Event tables for the HbA1c survival analyses |
| `conf_int/gap_120/step8_survival_plots_and_cox.py` | Kaplan–Meier curves, Cox models | Main-text survival figure; **Extended Data Table 2** (Cox hazard ratios vs normal glycemia) |
| `step8b_cox_threshold_summary_table.py` | `step8b_summary.md` | Cox threshold summary supporting Extended Data Table 2 |

### Exploratory note-derived assessments

Labelled exploratory in the Methods (*Exploratory note-derived clinical assessments*): PHQ-9,
general pain intensity, waist circumference, alcohol use, muscle strength.

| Script | Produces | Manuscript / supplementary item |
|---|---|---|
| `add_unstructured/run_trajectory_plots.py` | GEE trajectories per assessment domain | **Extended Data Table 4** and its figures |
| `add_unstructured/run_baseline_anchor_analysis.py` | change-from-baseline analyses | **Extended Data Table 4** |
| `add_unstructured/run_its_analysis.py` | interrupted-time-series pre/post | Supplementary note-derived results |
| `add_unstructured/run_elevated_analysis.py` | elevated-at-baseline subgroups | Supplementary note-derived results |
| `add_unstructured/forest_point_estimates.py` | forest point estimates | **Extended Data Table 4** |
| `add_unstructured/run_time_varying_covar.py` | time-varying-covariate models | Supplementary note-derived sensitivity |
| `add_unstructured/select_spline_df_assessment.py` | per-domain spline df | Model selection for the above |
| `add_unstructured/run_for_gap.py` | per-gap driver for the above | — |

### Runners

| Runner | Cohort | Grid | Role |
|---|---|---|---|
| `rerun_conf_int_clean_full.sh` | all | `GAPS_WITH_548` (all-data, note-derived), `GAPS_STRUCTURED` (structured-only) | **Produced the manuscript numbers**, end to end |
| `conf_int/run_all_gaps_all_data.sh` | all-data | `GAPS_PRIMARY` | Per-gap re-runs of the CI step scripts |
| `conf_int/run_all_gaps_structured_only.sh` | structured-only | `GAPS_STRUCTURED` | Per-gap structured-only sensitivity pipeline |
| `add_unstructured/run_all_gaps.sh` | note-derived | `GAPS_WITH_548` | Per-gap exploratory note-derived analyses |

---

## 3. Removed as unused

Every code path deleted in v2, with the reason, so that no reader has to judge for themselves
whether an unreferenced routine contributed to a reported result. None of these was reachable
from any reported result; the first four were never called from anywhere in the package.

| Removed | Where it was | Why it is gone |
|---|---|---|
| `_apply_adherence_censoring` | `step1_prepare_analysis_dataset.py` (~178–229) | A third variant of the follow-up rule, defined and never called from anywhere. Identified by the reviewer. |
| `_first_nonadherence_gap` | `step1_prepare_analysis_dataset.py` (~100) | Helper reachable only from `_apply_adherence_censoring`; dead once that was removed. |
| `_compute_adherence_flag` (singular) | `step1_prepare_analysis_dataset_a1c.py` (~151) | A fourth variant, using a nearest-mention test (`|d − m| ≤ g`, forward as well as backward) rather than the backward-only rule. Defined and never called. Not among the variants the reviewer identified; found while consolidating. |
| `GLP1_INJECTABLE_NAMES` | `step1_prepare_analysis_dataset.py` (~13) | A 6-name set that nothing read, sitting alongside inline regexes in the live cohort filter that listed the same agents. Two lists agreeing by coincidence. Replaced by `GLP1_BRAND_NAMES` / `GLP1_INGREDIENT_NAMES`, which the filter patterns are now built from, so the cohort definition has one source. |
| `_compute_censor_map_from_step8f` | `step1_prepare_analysis_dataset.py` (~113) | Live, but only via step0a: the second of the two follow-up rules. Replaced by `persistence.censor_days`. |
| `_compute_adherence_flags` (weight and HbA1c copies) | both step1 files | Live: the rule that produced the manuscript, duplicated line-for-line across the two files. Both replaced by the single `persistence.adherence_flags`. |
| local `_load_spline_df`, local `df_spline = 3` fallbacks | 10 scripts | Replaced by `model_spec.load_spline_df`, which raises instead of substituting a default model specification. |

---

## 4. Persistence-of-therapy grids

`gap_grids.sh` is the single source of truth; all four runners source it. In the submitted
package each runner declared its own list and they disagreed.

| Grid | Values | What it is |
|---|---|---|
| `GAPS_PRIMARY` | 30, 60, 90, 120, 150, 180, 365, 730 | The **eight** thresholds the manuscript reports. Primary analyses use the prespecified g = 120. |
| `GAPS_WITH_548` | the eight plus **548** | 548 (~18 months) is **not** a reported threshold. It appears only in supplementary figure sets and in the sample-size grid. It is retained deliberately, not dropped, because the submitted supplementary figures were produced with it. |
| `GAPS_STRUCTURED` | 30, 60, 90, 120, 150, 180, 365, 730 | The structured-only sensitivity cohort, which has never included 548 in either the submitted or the v2 package. |

Supplementary Table 1 keeps its original 8-threshold layout, as the supplement notes; the
attrition table is nonetheless computed over all nine and both layouts reproduce the published
values exactly.

## 5. Follow-up horizon caps

Three numbers look inconsistent across scripts. They are all "18 months or beyond" under
different month conventions, not different analysis windows, and each is documented where it
is defined. No estimate is reported beyond 18 months.

| Cap | Where | What it bounds |
|---|---|---|
| 730 | `MAX_DAYS_DEFAULT`, `step1_*.py --max-days` | Outer bound on the data step1 emits — wide enough that the 730-day persistence sensitivity analysis has data to use. Not an analysis window. |
| 548 | `step6b/step6c --max-days`, `step6d --truncate-days` | ~18 months as a calendar figure: contrast, plot and fitting horizon. |
| 540 | `MAX_FOLLOWUP_DAYS`, `step8_survival_time_to_*.py` | 18 × 30 days: event-time cap, so no time-to-event exceeds 18 months. |

## 6. Confidence level

Every interval derives its multiplier from one configurable confidence level
(`analysis_config.py`) rather than a hard-coded `1.96`. The default is 0.95, giving
1.959963984540054. Override for a whole run without editing code:

```bash
CI_CONF_LEVEL=0.99 bash code/rerun_conf_int_clean_full.sh
```

At the default level this changes confidence bounds by at most ~1.2 × 10⁻⁴ percentage points
relative to the submitted `1.96` — three orders of magnitude below the precision at which
results are reported, and identical after rounding to two decimal places. Point estimates,
standard errors and p-values are unaffected. See `CHANGE_LIST.md`.

## 7. Failure behaviour of the runners

The runners stop at the first failing step by default. The resume feature is opt-in
(`--resume`) and keys on a completion marker written only after a step exits 0, so a step that
crashed halfway is re-run rather than treated as finished.

```bash
bash code/rerun_conf_int_clean_full.sh                 # full run, stop on first error
bash code/rerun_conf_int_clean_full.sh --resume        # skip steps that completed
bash code/rerun_conf_int_clean_full.sh --keep-going    # continue past failures
```

## 8. Tests

```bash
python3 code/tests/test_persistence_and_step0a.py
```

Covers the patient-identifier dtype defect, the equivalence of the consolidated follow-up
rule's two readings, the explicit-stop and abutting-interval boundaries, the `baseline_carried_to_day0`
marker, the derived confidence multiplier, the fail-closed metformin helper and the absent
exclusion-column default.

## Data availability

The analyses use de-identified real-world EHR data (Sidus Insights / Harris) accessed under
data-use agreements; the source data are not included and cannot be redistributed. The code
contains no credentials or protected health information.

## Licence

MIT. See `LICENSE`.

The licence covers this analysis code only. It does not extend to the RespondHealth extraction
platform, which is proprietary and not distributed here, nor to the underlying electronic health
record data, which are licensed from Harris Computer Systems.

## Citing this code

Please cite the paper rather than the repository. `CITATION.cff` carries the machine-readable
record, which GitHub renders as a "Cite this repository" link.
