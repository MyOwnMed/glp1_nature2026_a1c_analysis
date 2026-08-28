#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# FULL RE-RUN: output/conf_int → output/conf_int_clean
#
# Uses the weight-cleaned CSV as source:
#   root_data/merged/step8g_with_unstructured_flags_with_assessments_weightcleaned.csv
#
# Reproduces ALL analysis types:
#   - 1_no_adherence + 1_no_adherence_full  (assessment domains, no gap censoring)
#   - add_unstructured/*  for gaps 30,60,90,120,150,180,365,548,730
#   - all_data/*          for gaps 30,60,90,120,150,180,365,730
#   - structured_only/*   for gaps 30,60,90,120,150,180,365,730
#   - spline_selection
#   - summary PDFs
#
# Phases:
#   0. Run step1 (weight + A1C + structured-only) with cleaned CSV
#   1. Run prepare_assessment_data (1_no_adherence base data)
#   2. Run add_unstructured gap analyses
#   3. Run all_data CI analyses
#   4. Run structured_only CI analyses
#   5. Spline selection + summary PDFs
#
# Usage:
#   nohup bash code/rerun_conf_int_clean_full.sh > output/conf_int_clean/nohup.log 2>&1 &
#
#   (no flag needed: a full, fail-fast run is the default)
#
#   --resume       skip steps that recorded a completion marker on an earlier run.
#                  Steps whose output directory has content but NO marker are
#                  re-run, because that is what a half-finished step looks like.
#   --keep-going   do not stop at the first failing step. Off by default.
#   --force        accepted and ignored; a full run is now the default. Kept so
#                  existing invocation strings keep working.
#
# ═══════════════════════════════════════════════════════════════════════════════
#
# ── PART II v2 CHANGES (code-review items 6 and 7) ────────────────────────────
#
# Item 6 — the runner logged failures and kept going.
#   * `set -e` is now set, so an unhandled command failure aborts the run.
#   * run_step checks the exit code of every step and stops the run by default.
#     Previously it computed an exit code that no caller looked at, so a failed
#     step1 was logged as FAILED and the pipeline went on to fit models against
#     stale or absent inputs.
#   * The resume feature is opt-in (--resume) rather than the default, and it now
#     keys on a completion marker written only after a step exits 0. Previously
#     any step whose output directory merely existed and was non-empty was
#     treated as complete, so a step that crashed halfway was skipped on the next
#     run unless --force was passed. A half-finished step is now detectable: the
#     directory has content but no marker, and the step is re-run.
#   * Missing required inputs are fatal, not a skip-with-warning.
#
# Item 7 — the gap grids disagreed between runners. All four runners now source
#   gap_grids.sh, the single source of truth. See that file for what each grid is
#   and why the 548-day threshold belongs to some of them and not others, and
#   README.md for the script-to-deliverable map.
#
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail
IFS=$' \n\t'

# ── Configuration ─────────────────────────────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CDIR="$ROOT_DIR/code"
CICODE="$ROOT_DIR/code/conf_int/gap_120"     # CI scripts (any gap)

PY=${PYTHON_BIN:-python3}
export MPLBACKEND=${MPLBACKEND:-Agg}

CLEAN_CSV="$ROOT_DIR/root_data/merged/step8g_with_unstructured_flags_with_assessments_weightcleaned.csv"
CLEAN_DIR="$ROOT_DIR/output/conf_int_clean"

# Single source of truth for the persistence grids (item 7).
# shellcheck source=gap_grids.sh
source "$(dirname "$0")/gap_grids.sh"
GAPS_UNSTRUCTURED=("${GAPS_WITH_548[@]}")
GAPS_ALL_DATA=("${GAPS_WITH_548[@]}")
# GAPS_STRUCTURED and MAX_DAYS come from gap_grids.sh unchanged.

RESUME=0
KEEP_GOING=0
for arg in "$@"; do
  case "$arg" in
    --resume)     RESUME=1 ;;
    --keep-going) KEEP_GOING=1 ;;
    --force)      : ;;  # full run is the default now; accepted for compatibility
    *) echo "Unknown argument: $arg" >&2
       echo "Usage: $0 [--resume] [--keep-going] [--force]" >&2
       exit 2 ;;
  esac
done

mkdir -p "$CLEAN_DIR"
MASTER_LOG="$CLEAN_DIR/rerun_master.log"
touch "$MASTER_LOG"

FAILED_STEPS=()

# ── Helpers ───────────────────────────────────────────────────────────────────
_ts() { date +"%Y-%m-%d %H:%M:%S"; }

tlog() {
  echo "[$(_ts)] $*" | tee -a "$MASTER_LOG"
}

die() {
  tlog "ABORTING: $*"
  tlog "The run stopped at the first failure. Pass --keep-going to continue past"
  tlog "failures, or --resume to skip steps that completed on an earlier run."
  exit 1
}

# Record a failure. Stops the run unless --keep-going was passed.
fail_step() {
  local name="$1"; local detail="$2"
  FAILED_STEPS+=("$name")
  tlog "${name} FAILED — ${detail}"
  if [[ $KEEP_GOING -eq 0 ]]; then
    die "${name} failed (${detail})"
  fi
  tlog "--keep-going set; continuing despite the failure in ${name}"
  return 0
}

# Completion marker for a step's output directory. Written only on exit 0, so its
# presence means "this step finished", not "this directory has files in it".
_marker() { echo "$1/.step_complete"; }

# should_skip OUTCHECK_DIR NAME -> 0 to skip, 1 to run
should_skip() {
  local outcheck="$1"; local name="$2"
  [[ $RESUME -eq 1 ]] || return 1
  if [[ -f "$(_marker "$outcheck")" ]]; then
    tlog "SKIP ${name} (completed $(cat "$(_marker "$outcheck")" 2>/dev/null))"
    return 0
  fi
  if [[ -d "$outcheck" ]] && [[ -n "$(ls -A "$outcheck" 2>/dev/null)" ]]; then
    tlog "RERUN ${name} (output present but no completion marker — half-finished)"
  fi
  return 1
}

mark_complete() {
  local outcheck="$1"
  mkdir -p "$outcheck"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$(_marker "$outcheck")"
}

# run_step NAME OUTCHECK_DIR LOG_FILE PYTHON_SCRIPT [ARGS...]
run_step() {
  local name="$1"; shift
  local outcheck="$1"; shift
  local log="$1"; shift
  local cmd=("$PY" "$@")

  if should_skip "$outcheck" "$name"; then
    return 0
  fi

  tlog "START ${name}"
  mkdir -p "$(dirname "$log")" "$outcheck"
  rm -f "$(_marker "$outcheck")"

  local rc=0
  ( cd "$ROOT_DIR" && "${cmd[@]}" ) >> "$log" 2>&1 || rc=$?
  if [[ $rc -ne 0 ]]; then
    fail_step "$name" "exit ${rc}; see $log"
    return 0
  fi
  mark_complete "$outcheck"
  tlog "${name} OK"
  return 0
}

# run_env_step NAME OUTCHECK_DIR LOG_FILE ENV_ASSIGNMENTS... -- COMMAND...
# Same contract as run_step, for steps that need environment overrides.
run_env_step() {
  local name="$1"; shift
  local outcheck="$1"; shift
  local log="$1"; shift
  local envs=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do envs+=("$1"); shift; done
  shift  # drop the --

  if should_skip "$outcheck" "$name"; then
    return 0
  fi

  tlog "START ${name}"
  mkdir -p "$(dirname "$log")" "$outcheck"
  rm -f "$(_marker "$outcheck")"

  local rc=0
  ( cd "$ROOT_DIR" && env "${envs[@]}" "$@" ) >> "$log" 2>&1 || rc=$?
  if [[ $rc -ne 0 ]]; then
    fail_step "$name" "exit ${rc}; see $log"
    return 0
  fi
  mark_complete "$outcheck"
  tlog "${name} OK"
  return 0
}

# ═══════════════════════════════════════════════════════════════════════════════
tlog "══════════════════════════════════════════════════════════════"
tlog "  FULL RE-RUN: conf_int → conf_int_clean"
tlog "  Source CSV: $CLEAN_CSV"
tlog "  Output:     $CLEAN_DIR"
tlog "  Resume:     $RESUME   (0 = run everything)"
tlog "  Keep going: $KEEP_GOING   (0 = stop at first failure)"
tlog "══════════════════════════════════════════════════════════════"
echo ""

# Verify source CSV exists
if [[ ! -f "$CLEAN_CSV" ]]; then
  tlog "ERROR: Source CSV not found: $CLEAN_CSV"
  exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0: Step 1 data preparation
# ═══════════════════════════════════════════════════════════════════════════════
tlog "── Phase 0: Step 1 data preparation ──"

STEP1_WEIGHT="$CLEAN_DIR/step1_weight"
STEP1_A1C="$CLEAN_DIR/step1_a1c"

# 0a. Step 1 — weight (all gaps)
run_step "step1_weight_all_gaps" \
  "$STEP1_WEIGHT" \
  "$CLEAN_DIR/logs/step1_weight.log" \
  "$CDIR/step1_prepare_analysis_dataset.py" \
  --input-csv "$CLEAN_CSV" \
  --outdir "$STEP1_WEIGHT" \
  --max-days "$MAX_DAYS" \
  --adherence-gaps ${GAPS_UNSTRUCTURED[*]}

# 0b. Step 1 — A1C (all gaps)
run_step "step1_a1c_all_gaps" \
  "$STEP1_A1C" \
  "$CLEAN_DIR/logs/step1_a1c.log" \
  "$CDIR/step1_prepare_analysis_dataset_a1c.py" \
  --input-csv "$CLEAN_CSV" \
  --outdir "$STEP1_A1C" \
  --max-days "$MAX_DAYS" \
  --adherence-gaps ${GAPS_UNSTRUCTURED[*]}

# 0c. Structured-only prefilter
SO_PREFILTERED="$CLEAN_DIR/prefiltered_structured_only.csv"
SO_STEP1_WEIGHT="$CLEAN_DIR/structured_only_step1_weight"
SO_STEP1_A1C="$CLEAN_DIR/structured_only_step1_a1c"

# The prefilter writes a single CSV rather than a directory, so it carries its own
# marker beside the output instead of using run_step's directory marker.
SO_PREFILTER_MARKER="$CLEAN_DIR/.step_complete_structured_only_prefilter"
if [[ $RESUME -eq 1 ]] && [[ -f "$SO_PREFILTER_MARKER" ]] && [[ -f "$SO_PREFILTERED" ]]; then
  tlog "SKIP structured_only prefilter (completed $(cat "$SO_PREFILTER_MARKER"))"
else
  if [[ -f "$SO_PREFILTERED" ]] && [[ ! -f "$SO_PREFILTER_MARKER" ]]; then
    tlog "RERUN structured_only prefilter (output present but no completion marker)"
  fi
  tlog "START structured_only prefilter"
  mkdir -p "$CLEAN_DIR/logs"
  rm -f "$SO_PREFILTER_MARKER"
  rc=0
  ( cd "$ROOT_DIR" && $PY "$CDIR/structured_only/step0_prefilter_raw.py" \
      --input-csv "$CLEAN_CSV" \
      --output-csv "$SO_PREFILTERED" ) >> "$CLEAN_DIR/logs/step0_prefilter.log" 2>&1 || rc=$?
  if [[ $rc -ne 0 ]]; then
    fail_step "structured_only prefilter" "exit $rc; see $CLEAN_DIR/logs/step0_prefilter.log"
  else
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$SO_PREFILTER_MARKER"
    tlog "structured_only prefilter OK"
  fi
fi

# Every structured-only step below reads this file. Missing it is fatal, not a
# warning: the phase would otherwise fit models against absent inputs.
if [[ ! -f "$SO_PREFILTERED" ]]; then
  die "structured_only prefilter produced no output: $SO_PREFILTERED"
fi

# 0d. Step 1 structured-only — weight
run_step "step1_structured_weight" \
  "$SO_STEP1_WEIGHT" \
  "$CLEAN_DIR/logs/step1_structured_weight.log" \
  "$CDIR/step1_prepare_analysis_dataset.py" \
  --input-csv "$SO_PREFILTERED" \
  --outdir "$SO_STEP1_WEIGHT" \
  --max-days "$MAX_DAYS" \
  --adherence-gaps ${GAPS_STRUCTURED[*]}

# 0e. Step 1 structured-only — A1C
run_step "step1_structured_a1c" \
  "$SO_STEP1_A1C" \
  "$CLEAN_DIR/logs/step1_structured_a1c.log" \
  "$CDIR/step1_prepare_analysis_dataset_a1c.py" \
  --input-csv "$SO_PREFILTERED" \
  --outdir "$SO_STEP1_A1C" \
  --max-days "$MAX_DAYS" \
  --adherence-gaps ${GAPS_STRUCTURED[*]}

echo ""
tlog "Phase 0 complete."
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Assessment domain base data (1_no_adherence)
# ═══════════════════════════════════════════════════════════════════════════════
tlog "── Phase 1: prepare_assessment_data (1_no_adherence) ──"

PHASE1_OUT="$CLEAN_DIR/1_no_adherence/data"
run_env_step "phase1_base_prepare_assessment_data" "$PHASE1_OUT" \
  "$CLEAN_DIR/logs/phase1_base.log" \
  "AU_DATADIR=$PHASE1_OUT" -- \
  "$PY" "$CDIR/add_unstructured/prepare_assessment_data.py"

# Run the analysis on the uncensored data (ITS, CFB, etc.)
tlog "Running 1_no_adherence analyses..."
NONADH_OUT="$CLEAN_DIR/1_no_adherence"
for script in run_its_analysis.py run_elevated_analysis.py run_trajectory_plots.py \
              run_baseline_anchor_analysis.py forest_point_estimates.py run_time_varying_covar.py; do
  sname="1_no_adherence_${script%.py}"
  run_env_step "$sname" "$NONADH_OUT/${script%.py}" \
    "$CLEAN_DIR/logs/${sname}.log" \
    "AU_DATADIR=$PHASE1_OUT" "AU_OUTROOT=$NONADH_OUT" -- \
    "$PY" "$CDIR/add_unstructured/$script"
done

# 1_no_adherence_full (same data, same analyses, second output set)
PHASE1F_OUT="$CLEAN_DIR/1_no_adherence_full/data"
run_env_step "phase1_full_prepare_assessment_data" "$PHASE1F_OUT" \
  "$CLEAN_DIR/logs/phase1_full.log" \
  "AU_DATADIR=$PHASE1F_OUT" -- \
  "$PY" "$CDIR/add_unstructured/prepare_assessment_data.py"

# Run the analysis on full uncensored data
tlog "Running 1_no_adherence_full analyses..."
NONADHF_OUT="$CLEAN_DIR/1_no_adherence_full"
for script in run_its_analysis.py run_elevated_analysis.py run_trajectory_plots.py \
              run_baseline_anchor_analysis.py forest_point_estimates.py run_time_varying_covar.py; do
  sname="1_no_adherence_full_${script%.py}"
  run_env_step "$sname" "$NONADHF_OUT/${script%.py}" \
    "$CLEAN_DIR/logs/${sname}.log" \
    "AU_DATADIR=$PHASE1F_OUT" "AU_OUTROOT=$NONADHF_OUT" -- \
    "$PY" "$CDIR/add_unstructured/$script"
done

echo ""
tlog "Phase 1 complete."
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: add_unstructured for all gaps
# ═══════════════════════════════════════════════════════════════════════════════
tlog "── Phase 2: add_unstructured gap analyses ──"

# run_for_gap.py has its own --force flag for its internal step skipping. A full
# run is now the default here, so pass it unless --resume was requested.
FORCE_FLAG="--force"
[[ $RESUME -eq 1 ]] && FORCE_FLAG=""

for gap in "${GAPS_UNSTRUCTURED[@]}"; do
  run_env_step "gap${gap}_add_unstructured" "$CLEAN_DIR/gap_${gap}/add_unstructured" \
    "$CLEAN_DIR/logs/gap${gap}_add_unstructured.log" \
    "CONF_INT_DIR=$CLEAN_DIR" -- \
    "$PY" "$CDIR/add_unstructured/run_for_gap.py" --gap "$gap" ${FORCE_FLAG:+$FORCE_FLAG}
done

echo ""
tlog "Phase 2 complete."
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: all_data CI analyses for all gaps
# ═══════════════════════════════════════════════════════════════════════════════
tlog "── Phase 3: all_data CI analyses ──"

run_gap_all_data() {
  local GAP="$1"
  local OUT="$CLEAN_DIR/gap_${GAP}/all_data"
  local LOG_DIR="$OUT/run_logs"
  mkdir -p "$LOG_DIR"

  local WEIGHT_CSV="$STEP1_WEIGHT/analysis_ready_gap${GAP}.csv"
  local A1C_CSV="$STEP1_A1C/analysis_ready_a1c_gap${GAP}.csv"
  # Use model configs from the MAIN pipeline (spline df choice is structural)
  local WEIGHT_CONFIG="$ROOT_DIR/output/gap_${GAP}/step2_select_spline_df/model_config.json"
  local A1C_CONFIG="$ROOT_DIR/output/gap_${GAP}/step2_select_spline_df_a1c/model_config_a1c.json"

  local WEIGHT_EVENTS="$OUT/step8_survival_time_to_weight_loss/step8_weight_time_to_threshold_events.csv"
  local A1C_EVENTS="$OUT/step8_survival_time_to_a1c_drop/step8_a1c_time_to_threshold_events.csv"

  tlog "=== all_data GAP=${GAP} START ==="

  # Validate inputs. A missing step1 output is fatal: continuing would fit models
  # against absent inputs, which is exactly what item 6 was about.
  for f in "$WEIGHT_CSV" "$A1C_CSV"; do
    if [[ ! -f "$f" ]]; then
      die "Required input not found for gap ${GAP} all_data: $f (step1 did not produce it)"
    fi
  done
  # Model configs are also a hard requirement downstream: the spline degrees of
  # freedom are a model specification, and step5/step4 now refuse to default them.
  for f in "$WEIGHT_CONFIG" "$A1C_CONFIG"; do
    if [[ ! -f "$f" ]]; then
      die "Model config not found for gap ${GAP} all_data: $f (run step2 for this gap first)"
    fi
  done

  # Step 4: Predictive plots
  run_step "g${GAP}_ad_step4_pred_wt" "$OUT/step4_predictive_plots" \
    "$LOG_DIR/step4_predictive_weight.log" \
    "$CICODE/step4_predictive_plots.py" \
    --input-csv "$WEIGHT_CSV" --config-json "$WEIGHT_CONFIG" \
    --outdir "$OUT/step4_predictive_plots" --adherence-gap-days "$GAP"

  run_step "g${GAP}_ad_step4_pred_a1c" "$OUT/step4_predictive_plots_a1c" \
    "$LOG_DIR/step4_predictive_a1c.log" \
    "$CICODE/step4_predictive_plots_a1c.py" \
    --input-csv "$A1C_CSV" --config-json "$A1C_CONFIG" \
    --outdir "$OUT/step4_predictive_plots_a1c" --adherence-gap-days "$GAP"

  # Step 4: Observed summaries
  run_step "g${GAP}_ad_step4_obs" "$OUT/step4_observed_summary_plots" \
    "$LOG_DIR/step4_observed_summaries.log" \
    "$CICODE/step4_observed_summary_plots.py" \
    --weight-csv "$WEIGHT_CSV" --a1c-csv "$A1C_CSV" \
    --outdir "$OUT/step4_observed_summary_plots" \
    --adherence-gap-days "$GAP" --max-days "$MAX_DAYS" --bin-width 90

  # Step 6: Stratified by covariates
  run_step "g${GAP}_ad_step6_wt" "$OUT/step6_stratified_by_covariates_weight" \
    "$LOG_DIR/step6_stratified_weight.log" \
    "$CICODE/step6_stratified_by_covariates_weight.py" \
    --input-csv "$WEIGHT_CSV" --config-json "$WEIGHT_CONFIG" \
    --outdir "$OUT/step6_stratified_by_covariates_weight" --adherence-gap-days "$GAP"

  run_step "g${GAP}_ad_step6_a1c" "$OUT/step6_stratified_by_covariates_a1c" \
    "$LOG_DIR/step6_stratified_a1c.log" \
    "$CICODE/step6_stratified_by_covariates_a1c.py" \
    --input-csv "$A1C_CSV" --config-json "$A1C_CONFIG" \
    --outdir "$OUT/step6_stratified_by_covariates_a1c" --adherence-gap-days "$GAP"

  # Step 6cc: 3-way stratification
  run_step "g${GAP}_ad_step6cc_wt" "$OUT/step6cc_3way_by_age_sex_weight" \
    "$LOG_DIR/step6cc_3way_weight.log" \
    "$CICODE/step6cc_3waystrat_covariates_weight.py" \
    --input-csv "$WEIGHT_CSV" --outdir "$OUT/step6cc_3way_by_age_sex_weight" \
    --adherence-gap-days "$GAP" --max-days "$MAX_DAYS"

  run_step "g${GAP}_ad_step6cc_a1c" "$OUT/step6cc_3way_by_age_sex_a1c" \
    "$LOG_DIR/step6cc_3way_a1c.log" \
    "$CICODE/step6cc_3waystrat_covariates_a1c.py" \
    --input-csv "$A1C_CSV" --outdir "$OUT/step6cc_3way_by_age_sex_a1c" \
    --adherence-gap-days "$GAP" --max-days "$MAX_DAYS"

  # Step 8: Time-to-event CSVs
  run_step "g${GAP}_ad_step8_tte_wt" "$OUT/step8_survival_time_to_weight_loss" \
    "$LOG_DIR/step8_time_to_weight.log" \
    "$CDIR/step8_survival_time_to_weight_loss.py" \
    --input-csv "$WEIGHT_CSV" --outdir "$OUT/step8_survival_time_to_weight_loss" \
    --adherence-gap-days "$GAP"

  run_step "g${GAP}_ad_step8_tte_a1c" "$OUT/step8_survival_time_to_a1c_drop" \
    "$LOG_DIR/step8_time_to_a1c.log" \
    "$CDIR/step8_survival_time_to_a1c_drop.py" \
    --input-csv "$A1C_CSV" --outdir "$OUT/step8_survival_time_to_a1c_drop" \
    --adherence-gap-days "$GAP"

  # Step 8: Survival plots + Cox
  run_step "g${GAP}_ad_step8_surv" "$OUT/step8_survival_plots_and_cox" \
    "$LOG_DIR/step8_survival.log" \
    "$CICODE/step8_survival_plots_and_cox.py" \
    --weight-events-csv "$WEIGHT_EVENTS" --a1c-events-csv "$A1C_EVENTS" \
    --analysis-a1c-csv "$A1C_CSV" --analysis-weight-csv "$WEIGHT_CSV" \
    --outdir-base "$OUT/step8_survival_plots_and_cox" \
    --figdir-base "$OUT/step8_survival_plots_and_cox/plots" \
    --adherence-gap-days "$GAP"

  tlog "=== all_data GAP=${GAP} DONE ==="
}

for G in "${GAPS_ALL_DATA[@]}"; do
  run_gap_all_data "$G"
done

echo ""
tlog "Phase 3 complete."
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: structured_only for all gaps (full pipeline: steps 0-8)
# ═══════════════════════════════════════════════════════════════════════════════
tlog "── Phase 4: structured_only analyses ──"

run_gap_structured() {
  local GAP="$1"
  local WEIGHT_CSV="$SO_STEP1_WEIGHT/analysis_ready_gap${GAP}.csv"
  local A1C_CSV="$SO_STEP1_A1C/analysis_ready_a1c_gap${GAP}.csv"

  local OUT="$CLEAN_DIR/gap_${GAP}/structured_only"
  local LOG_DIR="$OUT/run_logs"
  mkdir -p "$LOG_DIR"

  local WEIGHT_CONFIG="$OUT/step2_select_spline_df/model_config.json"
  local A1C_CONFIG="$OUT/step2_select_spline_df_a1c/model_config_a1c.json"
  local WEIGHT_EVENTS="$OUT/step8_survival_time_to_weight_loss/step8_weight_time_to_threshold_events.csv"
  local A1C_EVENTS="$OUT/step8_survival_time_to_a1c_drop/step8_a1c_time_to_threshold_events.csv"

  tlog "=== structured_only GAP=${GAP} START ==="

  # Validate inputs (fatal, per item 6).
  for f in "$WEIGHT_CSV" "$A1C_CSV"; do
    if [[ ! -f "$f" ]]; then
      die "Required input not found for gap ${GAP} structured_only: $f (step1 did not produce it)"
    fi
  done

  # Step 0: Population table
  run_step "g${GAP}_so_step0" "$OUT/step0_analysis_population_table" \
    "$LOG_DIR/step0_population_table.log" \
    "$CDIR/step0_analysis_population_table.py" \
    --input-csv "$WEIGHT_CSV" --outdir "$OUT/step0_analysis_population_table" \
    --adherence-gap-days "$GAP"

  # Step 2: Spline df selection
  run_step "g${GAP}_so_step2_wt" "$OUT/step2_select_spline_df" \
    "$LOG_DIR/step2_spline_weight.log" \
    "$CDIR/step2_select_spline_df.py" \
    --input-csv "$WEIGHT_CSV" --outdir "$OUT/step2_select_spline_df" \
    --adherence-gap-days "$GAP"

  run_step "g${GAP}_so_step2_a1c" "$OUT/step2_select_spline_df_a1c" \
    "$LOG_DIR/step2_spline_a1c.log" \
    "$CDIR/step2_select_spline_df_a1c.py" \
    --input-csv "$A1C_CSV" --outdir "$OUT/step2_select_spline_df_a1c"

  # Confirm model configs exist after step2.
  for cfg in "$WEIGHT_CONFIG" "$A1C_CONFIG"; do
    if [[ ! -f "$cfg" ]]; then
      die "Model config missing after step2 for gap ${GAP} structured_only: $cfg"
    fi
  done

  # Step 3: GEE baseline fit
  run_step "g${GAP}_so_step3_wt" "$OUT/step3_fit_gee_baseline" \
    "$LOG_DIR/step3_gee_weight.log" \
    "$CDIR/step3_fit_gee_baseline.py" \
    --input-csv "$WEIGHT_CSV" --config-json "$WEIGHT_CONFIG" \
    --outdir "$OUT/step3_fit_gee_baseline" --adherence-gap-days "$GAP"

  run_step "g${GAP}_so_step3_a1c" "$OUT/step3_fit_gee_baseline_a1c" \
    "$LOG_DIR/step3_gee_a1c.log" \
    "$CDIR/step3_fit_gee_baseline_a1c.py" \
    --input-csv "$A1C_CSV" --config-json "$A1C_CONFIG" \
    --outdir "$OUT/step3_fit_gee_baseline_a1c" --adherence-gap-days "$GAP"

  # Step 4: Predictive + observed plots (CI versions)
  run_step "g${GAP}_so_step4_wt" "$OUT/step4_predictive_plots" \
    "$LOG_DIR/step4_predictive_weight.log" \
    "$CICODE/step4_predictive_plots.py" \
    --input-csv "$WEIGHT_CSV" --config-json "$WEIGHT_CONFIG" \
    --outdir "$OUT/step4_predictive_plots" --adherence-gap-days "$GAP"

  run_step "g${GAP}_so_step4_a1c" "$OUT/step4_predictive_plots_a1c" \
    "$LOG_DIR/step4_predictive_a1c.log" \
    "$CICODE/step4_predictive_plots_a1c.py" \
    --input-csv "$A1C_CSV" --config-json "$A1C_CONFIG" \
    --outdir "$OUT/step4_predictive_plots_a1c" --adherence-gap-days "$GAP"

  run_step "g${GAP}_so_step4_obs" "$OUT/step4_observed_summary_plots" \
    "$LOG_DIR/step4_observed_summaries.log" \
    "$CICODE/step4_observed_summary_plots.py" \
    --weight-csv "$WEIGHT_CSV" --a1c-csv "$A1C_CSV" \
    --outdir "$OUT/step4_observed_summary_plots" \
    --adherence-gap-days "$GAP" --max-days "$MAX_DAYS" --bin-width 90

  # Step 5: Forest contrasts
  run_step "g${GAP}_so_step5_wt" "$OUT/step5_forest_contrasts_weight" \
    "$LOG_DIR/step5_forest_weight.log" \
    "$CDIR/step5_forest_contrasts_weight.py" \
    --input-csv "$WEIGHT_CSV" --config-json "$WEIGHT_CONFIG" \
    --outdir "$OUT/step5_forest_contrasts_weight" --adherence-gap-days "$GAP"

  run_step "g${GAP}_so_step5_a1c" "$OUT/step5_forest_contrasts_a1c" \
    "$LOG_DIR/step5_forest_a1c.log" \
    "$CDIR/step5_forest_contrasts_a1c.py" \
    --input-csv "$A1C_CSV" --config-json "$A1C_CONFIG" \
    --outdir "$OUT/step5_forest_contrasts_a1c" --adherence-gap-days "$GAP"

  # Step 6: Stratified by covariates (CI versions)
  run_step "g${GAP}_so_step6_wt" "$OUT/step6_stratified_by_covariates_weight" \
    "$LOG_DIR/step6_stratified_weight.log" \
    "$CICODE/step6_stratified_by_covariates_weight.py" \
    --input-csv "$WEIGHT_CSV" --config-json "$WEIGHT_CONFIG" \
    --outdir "$OUT/step6_stratified_by_covariates_weight" --adherence-gap-days "$GAP"

  run_step "g${GAP}_so_step6_a1c" "$OUT/step6_stratified_by_covariates_a1c" \
    "$LOG_DIR/step6_stratified_a1c.log" \
    "$CICODE/step6_stratified_by_covariates_a1c.py" \
    --input-csv "$A1C_CSV" --config-json "$A1C_CONFIG" \
    --outdir "$OUT/step6_stratified_by_covariates_a1c" --adherence-gap-days "$GAP"

  # Step 6b: Stratified contrasts
  run_step "g${GAP}_so_step6b_wt" "$OUT/step6b_stratified_contrasts_weight" \
    "$LOG_DIR/step6b_stratified_weight.log" \
    "$CDIR/step6b_stratified_contrasts_weight.py" \
    --input-csv "$WEIGHT_CSV" --config-json "$WEIGHT_CONFIG" \
    --outdir "$OUT/step6b_stratified_contrasts_weight" \
    --adherence-gap-days "$GAP" --time-days 365

  run_step "g${GAP}_so_step6b_a1c" "$OUT/step6b_stratified_contrasts_a1c" \
    "$LOG_DIR/step6b_stratified_a1c.log" \
    "$CDIR/step6b_stratified_contrasts_a1c.py" \
    --input-csv "$A1C_CSV" --config-json "$A1C_CONFIG" \
    --outdir "$OUT/step6b_stratified_contrasts_a1c" \
    --adherence-gap-days "$GAP" --time-days 365

  # Step 6c: Stratified forest plots
  run_step "g${GAP}_so_step6c_wt" "$OUT/step6c_stratified_by_covariates_weight" \
    "$LOG_DIR/step6c_forest_weight.log" \
    "$CDIR/step6c_stratified_forest_plots_weight.py" \
    --input-csv "$WEIGHT_CSV" --config-json "$WEIGHT_CONFIG" \
    --outdir "$OUT/step6c_stratified_by_covariates_weight" \
    --outdir-main "$OUT/step6c_stratified_by_covariates_weight/main" \
    --adherence-gap-days "$GAP"

  run_step "g${GAP}_so_step6c_a1c" "$OUT/step6c_stratified_by_covariates_a1c" \
    "$LOG_DIR/step6c_forest_a1c.log" \
    "$CDIR/step6c_stratified_forest_plots.py" \
    --input-csv "$A1C_CSV" --config-json "$A1C_CONFIG" \
    --outdir "$OUT/step6c_stratified_by_covariates_a1c" \
    --outdir-main "$OUT/step6c_stratified_by_covariates_a1c/main" \
    --adherence-gap-days "$GAP"

  # Step 6cc: 3-way stratification (CI versions)
  run_step "g${GAP}_so_step6cc_wt" "$OUT/step6cc_3way_by_age_sex_weight" \
    "$LOG_DIR/step6cc_3way_weight.log" \
    "$CICODE/step6cc_3waystrat_covariates_weight.py" \
    --input-csv "$WEIGHT_CSV" --outdir "$OUT/step6cc_3way_by_age_sex_weight" \
    --adherence-gap-days "$GAP" --max-days "$MAX_DAYS"

  run_step "g${GAP}_so_step6cc_a1c" "$OUT/step6cc_3way_by_age_sex_a1c" \
    "$LOG_DIR/step6cc_3way_a1c.log" \
    "$CICODE/step6cc_3waystrat_covariates_a1c.py" \
    --input-csv "$A1C_CSV" --outdir "$OUT/step6cc_3way_by_age_sex_a1c" \
    --adherence-gap-days "$GAP" --max-days "$MAX_DAYS"

  # Step 6d: GLP-1 group comparisons
  run_step "g${GAP}_so_step6d_wt" "$OUT/step6d_glp1_groups_weight" \
    "$LOG_DIR/step6d_glp1_weight.log" \
    "$CDIR/step6d_groups_by_glp1.py" \
    --outcome weight --input-csv "$WEIGHT_CSV" --config-json "$WEIGHT_CONFIG" \
    --outdir "$OUT/step6d_glp1_groups_weight" --adherence-gap-days "$GAP"

  run_step "g${GAP}_so_step6d_a1c" "$OUT/step6d_glp1_groups_a1c" \
    "$LOG_DIR/step6d_glp1_a1c.log" \
    "$CDIR/step6d_groups_by_glp1.py" \
    --outcome a1c --input-csv "$A1C_CSV" --config-json "$A1C_CONFIG" \
    --outdir "$OUT/step6d_glp1_groups_a1c" --adherence-gap-days "$GAP"

  # Step 8: Time-to-event CSVs
  run_step "g${GAP}_so_step8_tte_wt" "$OUT/step8_survival_time_to_weight_loss" \
    "$LOG_DIR/step8_time_to_weight.log" \
    "$CDIR/step8_survival_time_to_weight_loss.py" \
    --input-csv "$WEIGHT_CSV" --outdir "$OUT/step8_survival_time_to_weight_loss" \
    --adherence-gap-days "$GAP"

  run_step "g${GAP}_so_step8_tte_a1c" "$OUT/step8_survival_time_to_a1c_drop" \
    "$LOG_DIR/step8_time_to_a1c.log" \
    "$CDIR/step8_survival_time_to_a1c_drop.py" \
    --input-csv "$A1C_CSV" --outdir "$OUT/step8_survival_time_to_a1c_drop" \
    --adherence-gap-days "$GAP"

  # Step 8: Survival plots + Cox (CI version)
  run_step "g${GAP}_so_step8_surv" "$OUT/step8_survival_plots_and_cox" \
    "$LOG_DIR/step8_survival_plots.log" \
    "$CICODE/step8_survival_plots_and_cox.py" \
    --weight-events-csv "$WEIGHT_EVENTS" --a1c-events-csv "$A1C_EVENTS" \
    --analysis-a1c-csv "$A1C_CSV" --analysis-weight-csv "$WEIGHT_CSV" \
    --outdir-base "$OUT/step8_survival_plots_and_cox" \
    --figdir-base "$OUT/step8_survival_plots_and_cox/plots" \
    --adherence-gap-days "$GAP"

  # Step 8b: Cox threshold summary
  if should_skip "$OUT/step8b" "g${GAP}_so_step8b"; then
    :
  else
    tlog "START g${GAP}_so_step8b"
    local TMPBASE
    TMPBASE=$(mktemp -d)
    local TMPGAP="$TMPBASE/output/gap_${GAP}"
    mkdir -p "$TMPGAP"
    ln -s "$OUT/step8_survival_time_to_weight_loss" "$TMPGAP/step8_survival_time_to_weight_loss"
    ln -s "$OUT/step8_survival_time_to_a1c_drop"   "$TMPGAP/step8_survival_time_to_a1c_drop"
    ln -s "$OUT/step8_survival_plots_and_cox"       "$TMPGAP/step8_survival_plots_and_cox"
    mkdir -p "$OUT/step8b"
    local rc=0
    ( cd "$TMPBASE" && "$PY" "$CDIR/step8b_cox_threshold_summary_table.py" \
        --adherence-gap-days "$GAP" \
        --out-markdown "$OUT/step8b/step8b_summary.md" ) >> "$LOG_DIR/step8b.log" 2>&1 || rc=$?
    rm -rf "$TMPBASE"
    if [[ $rc -ne 0 ]]; then
      fail_step "g${GAP}_so_step8b" "exit $rc; see $LOG_DIR/step8b.log"
    else
      mark_complete "$OUT/step8b"
      tlog "g${GAP}_so_step8b OK"
    fi
  fi

  tlog "=== structured_only GAP=${GAP} DONE ==="
}

for G in "${GAPS_STRUCTURED[@]}"; do
  run_gap_structured "$G"
done

echo ""
tlog "Phase 4 complete."
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Spline selection + summary outputs
# ═══════════════════════════════════════════════════════════════════════════════
tlog "── Phase 5: Spline selection + summaries ──"

SPLINE_OUT="$CLEAN_DIR/spline_selection"
run_step "spline_selection" "$SPLINE_OUT" \
  "$CLEAN_DIR/logs/spline_selection.log" \
  "$CDIR/add_unstructured/select_spline_df_assessment.py" \
  --conf-int-dir "$CLEAN_DIR" \
  --out-dir "$SPLINE_OUT"

# Summary PDFs (build from new conf_int_clean data)
tlog "Building summary PDFs..."

# Elevated gaps PDF
if [[ -f "$CDIR/add_unstructured/build_gap_pdf.py" ]]; then
  tlog "START build_gap_pdf"
  rc=0
  ( cd "$ROOT_DIR" && $PY "$CDIR/add_unstructured/build_gap_pdf.py" \
      --conf-int-dir "$CLEAN_DIR" \
      --outfile "$CLEAN_DIR/add_unstructured_elevated_gaps.pdf" ) \
    >> "$CLEAN_DIR/logs/build_gap_pdf.log" 2>&1 || rc=$?
  if [[ $rc -ne 0 ]]; then
    fail_step "build_gap_pdf" "exit $rc; see $CLEAN_DIR/logs/build_gap_pdf.log"
  else
    tlog "build_gap_pdf OK"
  fi
else
  # Deliverable formatters are deliberately not distributed with the public
  # release (README, "Scope note"). Skip the step and say so, rather than
  # failing or dropping it silently.
  tlog "SKIP build_gap_pdf — add_unstructured/build_gap_pdf.py is not included in the public release; deliverable formatting is out of scope (see README 'Scope note')"
fi

# Comparison: all vs structured_only
if [[ -f "$CDIR/structured_only/build_comparison_pdf.py" ]]; then
  tlog "START build_comparison_pdf"
  rc=0
  ( cd "$ROOT_DIR" && $PY "$CDIR/structured_only/build_comparison_pdf.py" \
      --conf-int-dir "$CLEAN_DIR" \
      --outfile "$CLEAN_DIR/comparison_all_vs_structured_only.pdf" ) \
    >> "$CLEAN_DIR/logs/build_comparison_pdf.log" 2>&1 || rc=$?
  if [[ $rc -ne 0 ]]; then
    fail_step "build_comparison_pdf" "exit $rc; see $CLEAN_DIR/logs/build_comparison_pdf.log"
  else
    tlog "build_comparison_pdf OK"
  fi
else
  # Deliverable formatters are deliberately not distributed with the public
  # release (README, "Scope note"). Skip the step and say so, rather than
  # failing or dropping it silently.
  tlog "SKIP build_comparison_pdf — structured_only/build_comparison_pdf.py is not included in the public release; deliverable formatting is out of scope (see README 'Scope note')"
fi

# Comparison (select version)
if [[ -f "$CDIR/structured_only/build_comparison_pdf_select.py" ]]; then
  tlog "START build_comparison_pdf_select"
  rc=0
  ( cd "$ROOT_DIR" && $PY "$CDIR/structured_only/build_comparison_pdf_select.py" \
      --conf-int-dir "$CLEAN_DIR" \
      --outfile "$CLEAN_DIR/comparison_all_vs_structured_only_select.pdf" ) \
    >> "$CLEAN_DIR/logs/build_comparison_pdf_select.log" 2>&1 || rc=$?
  if [[ $rc -ne 0 ]]; then
    fail_step "build_comparison_pdf_select" "exit $rc; see $CLEAN_DIR/logs/build_comparison_pdf_select.log"
  else
    tlog "build_comparison_pdf_select OK"
  fi
else
  # Deliverable formatters are deliberately not distributed with the public
  # release (README, "Scope note"). Skip the step and say so, rather than
  # failing or dropping it silently.
  tlog "SKIP build_comparison_pdf_select — structured_only/build_comparison_pdf_select.py is not included in the public release; deliverable formatting is out of scope (see README 'Scope note')"
fi

# Freq/population summaries (need env override since they hardcode paths)
if [[ -f "$CDIR/add_unstructured/freq/mention_frequency.py" ]]; then
  tlog "START mention_frequency"
  # The freq scripts hardcode output/conf_int as OUTDIR.
  # We create a symlink temporarily so data dir resolves correctly, then move output.
  rc=0
  ( cd "$ROOT_DIR" && $PY -c "
import sys, importlib.util, os, types
os.chdir('$ROOT_DIR')
# Override the paths
import pandas as pd
from pathlib import Path
DATADIR = Path('$CLEAN_DIR/1_no_adherence/data')
OUTDIR = Path('$CLEAN_DIR')
OUTDIR.mkdir(parents=True, exist_ok=True)
# Load and patch the script
spec = importlib.util.spec_from_file_location('freq', '$CDIR/add_unstructured/freq/mention_frequency.py')
mod = importlib.util.module_from_spec(spec)
mod.DATADIR = DATADIR
mod.OUTDIR = OUTDIR
spec.loader.exec_module(mod)
" ) >> "$CLEAN_DIR/logs/mention_freq.log" 2>&1 || rc=$?
  if [[ $rc -ne 0 ]]; then
    fail_step "mention_frequency" "exit $rc; see $CLEAN_DIR/logs/mention_freq.log"
  else
    tlog "mention_frequency OK"
  fi
fi

if [[ -f "$CDIR/add_unstructured/freq/population_description.py" ]]; then
  tlog "START population_description"
  rc=0
  ( cd "$ROOT_DIR" && $PY -c "
import sys, importlib.util, os
os.chdir('$ROOT_DIR')
from pathlib import Path
DATADIR = Path('$CLEAN_DIR/1_no_adherence/data')
OUTDIR = Path('$CLEAN_DIR')
OUTDIR.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('popdesc', '$CDIR/add_unstructured/freq/population_description.py')
mod = importlib.util.module_from_spec(spec)
mod.DATA = DATADIR
mod.OUTDIR = OUTDIR
spec.loader.exec_module(mod)
" ) >> "$CLEAN_DIR/logs/population_desc.log" 2>&1 || rc=$?
  if [[ $rc -ne 0 ]]; then
    fail_step "population_description" "exit $rc; see $CLEAN_DIR/logs/population_desc.log"
  else
    tlog "population_description OK"
  fi
fi

echo ""
tlog "Phase 5 complete."
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════════════
# Reported from the tracked failure list rather than by grepping the master log
# for the word FAILED, which also matched log lines from previous runs.
tlog "══════════════════════════════════════════════════════════════"
if [[ ${#FAILED_STEPS[@]} -eq 0 ]]; then
  tlog "  FULL RE-RUN COMPLETE — all steps OK"
  tlog "  Results in: $CLEAN_DIR"
  tlog "  Master log: $MASTER_LOG"
  tlog "══════════════════════════════════════════════════════════════"
  exit 0
fi
tlog "  FULL RE-RUN FINISHED WITH ${#FAILED_STEPS[@]} FAILED STEP(S)"
tlog "  Results in: $CLEAN_DIR"
tlog "  Master log: $MASTER_LOG"
for s in "${FAILED_STEPS[@]}"; do tlog "    FAILED: $s"; done
tlog "  Exiting non-zero so a caller cannot mistake this for a clean run."
tlog "══════════════════════════════════════════════════════════════"
exit 1
