#!/usr/bin/env bash
# Run COMPLETE structured-only analysis pipeline for ALL adherence gaps.
#
# Mirrors the full output/gap_120 directory structure for each gap,
# using structured-only pre-filtered data from output/structured_only/step1_*/.
# Gap 120 is included but will be skipped (already complete).
#
# Uses CI scripts (code/conf_int/gap_120/*.py) where they exist,
# code/*.py for steps without CI versions (step0, step2, step3, step5,
# step6b, step6c, step6d, step8 TTE, step8b).
#
# Output: output/conf_int/gap_{GAP}/structured_only/ for each gap
#
# Usage:
#   bash code/conf_int/run_all_gaps_structured_only.sh [--force]
#
# --force : re-run all steps even if output already exists

# --force : re-run all steps even if output already exists

# ── PART II v2 (code-review items 6 and 7) ────────────────────────────────────
# Item 7: the gap grid is no longer declared here. All four runners source
#   gap_grids.sh, the single source of truth; see that file for what each grid is
#   and why 548 belongs to some and not others.
# Item 6: `set -e` is on, run_step checks every exit code and stops the run by
#   default, resume is opt-in via --resume and keys on a completion marker written
#   only on success, and missing required inputs are fatal rather than skipped.

set -euo pipefail

RESUME=0
KEEP_GOING=0
for arg in "$@"; do
  case "$arg" in
    --resume)     RESUME=1 ;;
    --keep-going) KEEP_GOING=1 ;;
    --force)      : ;;  # a full run is the default now; accepted for compatibility
    *) echo "Unknown argument: $arg" >&2
       echo "Usage: $0 [--resume] [--keep-going] [--force]" >&2
       exit 2 ;;
  esac
done

FAILED_STEPS=()
IFS=$' \n\t'

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

# Single source of truth for the persistence grids (item 7). The structured-only
# cohort has never included the 548-day threshold, in either the submitted or the
# v2 package, because its step1 data was built on the 8-threshold grid.
# shellcheck source=../gap_grids.sh
source "$ROOT_DIR/code/gap_grids.sh"
GAPS=("${GAPS_STRUCTURED[@]}")
CDIR="$ROOT_DIR/code"
CIDIR="$ROOT_DIR/code/conf_int/gap_120"   # CI scripts work for any gap
PY=${PYTHON_BIN:-python3}
export MPLBACKEND=${MPLBACKEND:-Agg}

_ts() { date +"%Y-%m-%d %H:%M:%S"; }


die() {
  echo "[$(_ts)] ABORTING: $*" | tee -a "${MASTER_LOG:-/dev/null}"
  exit 1
}

fail_step() {
  local name="$1"; local detail="$2"
  FAILED_STEPS+=("$name")
  echo "[$(_ts)] ${name} FAILED — ${detail}" | tee -a "${MASTER_LOG:-/dev/null}"
  if [[ $KEEP_GOING -eq 0 ]]; then
    die "${name} failed (${detail})"
  fi
  return 0
}

_marker() { echo "$1/.step_complete"; }

should_skip() {
  local outcheck="$1"; local name="$2"
  [[ $RESUME -eq 1 ]] || return 1
  if [[ -f "$(_marker "$outcheck")" ]]; then
    echo "[$(_ts)] SKIP ${name} (completed $(cat "$(_marker "$outcheck")" 2>/dev/null))"
    return 0
  fi
  if [[ -d "$outcheck" ]] && [[ -n "$(ls -A "$outcheck" 2>/dev/null)" ]]; then
    echo "[$(_ts)] RERUN ${name} (output present but no completion marker — half-finished)"
  fi
  return 1
}

mark_complete() {
  mkdir -p "$1"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$(_marker "$1")"
}

# ── Per-step runner: run unless a completion marker says it finished ──────────
run_step() {
  local name="$1"; shift
  local outcheck="$1"; shift
  local log="$1"; shift
  local cmd=("$PY" "$@")

  if should_skip "$outcheck" "$name"; then
    return 0
  fi

  echo "[$(_ts)] START ${name}" | tee -a "$MASTER_LOG" | tee "$log"
  mkdir -p "$(dirname "$log")" "$outcheck"
  rm -f "$(_marker "$outcheck")"

  local rc=0
  ( cd "$ROOT_DIR" && "${cmd[@]}" ) >> "$log" 2>&1 || rc=$?
  if [[ $rc -ne 0 ]]; then
    fail_step "$name" "exit ${rc}; see $log"
    return 0
  fi
  mark_complete "$outcheck"
  echo "[$(_ts)] ${name} OK" | tee -a "$MASTER_LOG"
  return 0
}

# ── Per-gap function ─────────────────────────────────────────────────────────
run_gap() {
  local GAP="$1"

  local SO_DATA="$ROOT_DIR/output/structured_only"
  local WEIGHT_CSV="$SO_DATA/step1_weight/analysis_ready_gap${GAP}.csv"
  local A1C_CSV="$SO_DATA/step1_a1c/analysis_ready_a1c_gap${GAP}.csv"

  local OUT="$ROOT_DIR/output/conf_int/gap_${GAP}/structured_only"
  local LOG_DIR="$OUT/run_logs_complete"
  mkdir -p "$LOG_DIR"

  MASTER_LOG="$LOG_DIR/MASTER.log"

  echo ""
  echo "========================================"
  echo "[$(_ts)] === STRUCTURED-ONLY GAP=${GAP} START ===" | tee -a "$MASTER_LOG"
  echo "========================================"

  # Validate inputs. Fatal, not a skip: continuing would fit models against
  # absent inputs (item 6).
  for f in "$WEIGHT_CSV" "$A1C_CSV"; do
    if [[ ! -f "$f" ]]; then
      die "Required input not found for gap ${GAP}: $f (structured-only step1 did not produce it)"
    fi
  done
  echo "[$(_ts)] Weight CSV: $WEIGHT_CSV" | tee -a "$MASTER_LOG"
  echo "[$(_ts)] A1C CSV   : $A1C_CSV"    | tee -a "$MASTER_LOG"

  local WEIGHT_CONFIG="$OUT/step2_select_spline_df/model_config.json"
  local A1C_CONFIG="$OUT/step2_select_spline_df_a1c/model_config_a1c.json"
  local WEIGHT_EVENTS="$OUT/step8_survival_time_to_weight_loss/step8_weight_time_to_threshold_events.csv"
  local A1C_EVENTS="$OUT/step8_survival_time_to_a1c_drop/step8_a1c_time_to_threshold_events.csv"

  # ── Step 0: Population table ────────────────────────────────────────
  run_step "step0_population_table" \
    "$OUT/step0_analysis_population_table" \
    "$LOG_DIR/step0_population_table.log" \
    "$CDIR/step0_analysis_population_table.py" \
    --input-csv "$WEIGHT_CSV" \
    --outdir "$OUT/step0_analysis_population_table" \
    --adherence-gap-days "$GAP"

  # ── Step 2: Spline df selection ─────────────────────────────────────
  run_step "step2_spline_weight" \
    "$OUT/step2_select_spline_df" \
    "$LOG_DIR/step2_spline_weight.log" \
    "$CDIR/step2_select_spline_df.py" \
    --input-csv "$WEIGHT_CSV" \
    --outdir "$OUT/step2_select_spline_df" \
    --adherence-gap-days "$GAP"

  run_step "step2_spline_a1c" \
    "$OUT/step2_select_spline_df_a1c" \
    "$LOG_DIR/step2_spline_a1c.log" \
    "$CDIR/step2_select_spline_df_a1c.py" \
    --input-csv "$A1C_CSV" \
    --outdir "$OUT/step2_select_spline_df_a1c"

  # Confirm model configs exist before continuing
  for cfg in "$WEIGHT_CONFIG" "$A1C_CONFIG"; do
    if [[ ! -f "$cfg" ]]; then
      echo "[$(_ts)] ERROR: Model config missing after step2: $cfg — aborting gap ${GAP}" | tee -a "$MASTER_LOG"
      return 1
    fi
  done

  # ── Step 3: GEE baseline fit ────────────────────────────────────────
  run_step "step3_gee_weight" \
    "$OUT/step3_fit_gee_baseline" \
    "$LOG_DIR/step3_gee_weight.log" \
    "$CDIR/step3_fit_gee_baseline.py" \
    --input-csv "$WEIGHT_CSV" \
    --config-json "$WEIGHT_CONFIG" \
    --outdir "$OUT/step3_fit_gee_baseline" \
    --adherence-gap-days "$GAP"

  run_step "step3_gee_a1c" \
    "$OUT/step3_fit_gee_baseline_a1c" \
    "$LOG_DIR/step3_gee_a1c.log" \
    "$CDIR/step3_fit_gee_baseline_a1c.py" \
    --input-csv "$A1C_CSV" \
    --config-json "$A1C_CONFIG" \
    --outdir "$OUT/step3_fit_gee_baseline_a1c" \
    --adherence-gap-days "$GAP"

  # ── Step 4: Predictive + observed plots (CI versions) ───────────────
  run_step "step4_predictive_weight" \
    "$OUT/step4_predictive_plots" \
    "$LOG_DIR/step4_predictive_weight.log" \
    "$CIDIR/step4_predictive_plots.py" \
    --input-csv "$WEIGHT_CSV" \
    --config-json "$WEIGHT_CONFIG" \
    --outdir "$OUT/step4_predictive_plots" \
    --adherence-gap-days "$GAP"

  run_step "step4_predictive_a1c" \
    "$OUT/step4_predictive_plots_a1c" \
    "$LOG_DIR/step4_predictive_a1c.log" \
    "$CIDIR/step4_predictive_plots_a1c.py" \
    --input-csv "$A1C_CSV" \
    --config-json "$A1C_CONFIG" \
    --outdir "$OUT/step4_predictive_plots_a1c" \
    --adherence-gap-days "$GAP"

  run_step "step4_observed_summaries" \
    "$OUT/step4_observed_summary_plots" \
    "$LOG_DIR/step4_observed_summaries.log" \
    "$CIDIR/step4_observed_summary_plots.py" \
    --weight-csv "$WEIGHT_CSV" \
    --a1c-csv "$A1C_CSV" \
    --outdir "$OUT/step4_observed_summary_plots" \
    --adherence-gap-days "$GAP" \
    --max-days "$MAX_DAYS" \
    --bin-width 90

  # ── Step 5: Forest contrasts ────────────────────────────────────────
  run_step "step5_forest_weight" \
    "$OUT/step5_forest_contrasts_weight" \
    "$LOG_DIR/step5_forest_weight.log" \
    "$CDIR/step5_forest_contrasts_weight.py" \
    --input-csv "$WEIGHT_CSV" \
    --config-json "$WEIGHT_CONFIG" \
    --outdir "$OUT/step5_forest_contrasts_weight" \
    --adherence-gap-days "$GAP"

  run_step "step5_forest_a1c" \
    "$OUT/step5_forest_contrasts_a1c" \
    "$LOG_DIR/step5_forest_a1c.log" \
    "$CDIR/step5_forest_contrasts_a1c.py" \
    --input-csv "$A1C_CSV" \
    --config-json "$A1C_CONFIG" \
    --outdir "$OUT/step5_forest_contrasts_a1c" \
    --adherence-gap-days "$GAP"

  # ── Step 6: Stratified by covariates (CI versions) ──────────────────
  run_step "step6_stratified_weight" \
    "$OUT/step6_stratified_by_covariates_weight" \
    "$LOG_DIR/step6_stratified_weight.log" \
    "$CIDIR/step6_stratified_by_covariates_weight.py" \
    --input-csv "$WEIGHT_CSV" \
    --config-json "$WEIGHT_CONFIG" \
    --outdir "$OUT/step6_stratified_by_covariates_weight" \
    --adherence-gap-days "$GAP"

  run_step "step6_stratified_a1c" \
    "$OUT/step6_stratified_by_covariates_a1c" \
    "$LOG_DIR/step6_stratified_a1c.log" \
    "$CIDIR/step6_stratified_by_covariates_a1c.py" \
    --input-csv "$A1C_CSV" \
    --config-json "$A1C_CONFIG" \
    --outdir "$OUT/step6_stratified_by_covariates_a1c" \
    --adherence-gap-days "$GAP"

  # ── Step 6b: Stratified contrasts ───────────────────────────────────
  run_step "step6b_stratified_weight" \
    "$OUT/step6b_stratified_contrasts_weight" \
    "$LOG_DIR/step6b_stratified_weight.log" \
    "$CDIR/step6b_stratified_contrasts_weight.py" \
    --input-csv "$WEIGHT_CSV" \
    --config-json "$WEIGHT_CONFIG" \
    --outdir "$OUT/step6b_stratified_contrasts_weight" \
    --adherence-gap-days "$GAP" \
    --time-days 365

  run_step "step6b_stratified_a1c" \
    "$OUT/step6b_stratified_contrasts_a1c" \
    "$LOG_DIR/step6b_stratified_a1c.log" \
    "$CDIR/step6b_stratified_contrasts_a1c.py" \
    --input-csv "$A1C_CSV" \
    --config-json "$A1C_CONFIG" \
    --outdir "$OUT/step6b_stratified_contrasts_a1c" \
    --adherence-gap-days "$GAP" \
    --time-days 365

  # ── Step 6c: Stratified forest plots ────────────────────────────────
  run_step "step6c_forest_weight" \
    "$OUT/step6c_stratified_by_covariates_weight" \
    "$LOG_DIR/step6c_forest_weight.log" \
    "$CDIR/step6c_stratified_forest_plots_weight.py" \
    --input-csv "$WEIGHT_CSV" \
    --config-json "$WEIGHT_CONFIG" \
    --outdir "$OUT/step6c_stratified_by_covariates_weight" \
    --outdir-main "$OUT/step6c_stratified_by_covariates_weight/main" \
    --adherence-gap-days "$GAP"

  run_step "step6c_forest_a1c" \
    "$OUT/step6c_stratified_by_covariates_a1c" \
    "$LOG_DIR/step6c_forest_a1c.log" \
    "$CDIR/step6c_stratified_forest_plots.py" \
    --input-csv "$A1C_CSV" \
    --config-json "$A1C_CONFIG" \
    --outdir "$OUT/step6c_stratified_by_covariates_a1c" \
    --outdir-main "$OUT/step6c_stratified_by_covariates_a1c/main" \
    --adherence-gap-days "$GAP"

  # ── Step 6cc: 3-way stratification (CI versions) ────────────────────
  run_step "step6cc_3way_weight" \
    "$OUT/step6cc_3way_by_age_sex_weight" \
    "$LOG_DIR/step6cc_3way_weight.log" \
    "$CIDIR/step6cc_3waystrat_covariates_weight.py" \
    --input-csv "$WEIGHT_CSV" \
    --outdir "$OUT/step6cc_3way_by_age_sex_weight" \
    --adherence-gap-days "$GAP" \
    --max-days "$MAX_DAYS"

  run_step "step6cc_3way_a1c" \
    "$OUT/step6cc_3way_by_age_sex_a1c" \
    "$LOG_DIR/step6cc_3way_a1c.log" \
    "$CIDIR/step6cc_3waystrat_covariates_a1c.py" \
    --input-csv "$A1C_CSV" \
    --outdir "$OUT/step6cc_3way_by_age_sex_a1c" \
    --adherence-gap-days "$GAP" \
    --max-days "$MAX_DAYS"

  # ── Step 6d: GLP-1 group comparisons ────────────────────────────────
  run_step "step6d_glp1_weight" \
    "$OUT/step6d_glp1_groups_weight" \
    "$LOG_DIR/step6d_glp1_weight.log" \
    "$CDIR/step6d_groups_by_glp1.py" \
    --outcome weight \
    --input-csv "$WEIGHT_CSV" \
    --config-json "$WEIGHT_CONFIG" \
    --outdir "$OUT/step6d_glp1_groups_weight" \
    --adherence-gap-days "$GAP"

  run_step "step6d_glp1_a1c" \
    "$OUT/step6d_glp1_groups_a1c" \
    "$LOG_DIR/step6d_glp1_a1c.log" \
    "$CDIR/step6d_groups_by_glp1.py" \
    --outcome a1c \
    --input-csv "$A1C_CSV" \
    --config-json "$A1C_CONFIG" \
    --outdir "$OUT/step6d_glp1_groups_a1c" \
    --adherence-gap-days "$GAP"

  # ── Step 8: Time-to-event CSVs (needed before plots + step8b) ───────
  run_step "step8_time_to_weight" \
    "$OUT/step8_survival_time_to_weight_loss" \
    "$LOG_DIR/step8_time_to_weight.log" \
    "$CDIR/step8_survival_time_to_weight_loss.py" \
    --input-csv "$WEIGHT_CSV" \
    --outdir "$OUT/step8_survival_time_to_weight_loss" \
    --adherence-gap-days "$GAP"

  run_step "step8_time_to_a1c" \
    "$OUT/step8_survival_time_to_a1c_drop" \
    "$LOG_DIR/step8_time_to_a1c.log" \
    "$CDIR/step8_survival_time_to_a1c_drop.py" \
    --input-csv "$A1C_CSV" \
    --outdir "$OUT/step8_survival_time_to_a1c_drop" \
    --adherence-gap-days "$GAP"

  # ── Step 8: Survival plots + Cox (CI version) ────────────────────────
  run_step "step8_survival_plots" \
    "$OUT/step8_survival_plots_and_cox" \
    "$LOG_DIR/step8_survival_plots.log" \
    "$CIDIR/step8_survival_plots_and_cox.py" \
    --weight-events-csv "$WEIGHT_EVENTS" \
    --a1c-events-csv "$A1C_EVENTS" \
    --analysis-a1c-csv "$A1C_CSV" \
    --analysis-weight-csv "$WEIGHT_CSV" \
    --outdir-base "$OUT/step8_survival_plots_and_cox" \
    --figdir-base "$OUT/step8_survival_plots_and_cox/plots" \
    --adherence-gap-days "$GAP"

  # ── Step 8b: Cox threshold summary (hardcoded path → symlink workaround) ──
  if [[ $FORCE -eq 0 ]] && [[ -d "$OUT/step8b" ]] && [[ -n "$(ls -A "$OUT/step8b" 2>/dev/null)" ]]; then
    echo "[$(_ts)] SKIP step8b (output exists)" | tee -a "$MASTER_LOG"
  else
    echo "[$(_ts)] START step8b" | tee -a "$MASTER_LOG"
    local STEP8B_LOG="$LOG_DIR/step8b.log"
    local TMPBASE
    TMPBASE=$(mktemp -d)
    local TMPGAP="$TMPBASE/output/gap_${GAP}"
    mkdir -p "$TMPGAP"
    ln -s "$OUT/step8_survival_time_to_weight_loss" "$TMPGAP/step8_survival_time_to_weight_loss"
    ln -s "$OUT/step8_survival_time_to_a1c_drop"   "$TMPGAP/step8_survival_time_to_a1c_drop"
    ln -s "$OUT/step8_survival_plots_and_cox"       "$TMPGAP/step8_survival_plots_and_cox"
    mkdir -p "$OUT/step8b"
    ( cd "$TMPBASE" && "$PY" "$CDIR/step8b_cox_threshold_summary_table.py" \
        --adherence-gap-days "$GAP" \
        --out-markdown "$OUT/step8b/step8b_summary.md" ) >> "$STEP8B_LOG" 2>&1
    local rc=$?
    rm -rf "$TMPBASE"
    if [[ $rc -ne 0 ]]; then
      echo "[$(_ts)] step8b FAILED (exit $rc)" | tee -a "$MASTER_LOG" | tee -a "$STEP8B_LOG"
    else
      echo "[$(_ts)] step8b OK" | tee -a "$MASTER_LOG"
    fi
  fi

  # ── Gap summary ──────────────────────────────────────────────────────
  # Reported from the tracked list rather than by grepping the master log for the
  # word FAILED, which also matched lines carried over from previous runs.
  if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
    echo "[$(_ts)] ⚠  ${#FAILED_STEPS[@]} step(s) FAILED so far — see $MASTER_LOG" | tee -a "$MASTER_LOG"
  else
    echo "[$(_ts)] === GAP=${GAP} all steps passed. Output: $OUT ===" | tee -a "$MASTER_LOG"
  fi
}

# ── Main: run all gaps ───────────────────────────────────────────────────────
echo "[$(_ts)] Starting structured-only analysis for gaps: ${GAPS[*]}"
for G in "${GAPS[@]}"; do
  run_gap "$G"
done

echo ""
if [[ ${#FAILED_STEPS[@]} -eq 0 ]]; then
  echo "[$(_ts)] All structured-only gaps complete — all steps OK."
else
  echo "[$(_ts)] Finished with ${#FAILED_STEPS[@]} failed step(s):"
  for s in "${FAILED_STEPS[@]}"; do echo "    FAILED: $s"; done
  exit 1
fi
