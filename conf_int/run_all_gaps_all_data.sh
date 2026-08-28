#!/usr/bin/env bash
# Run CI-enhanced plotting scripts for ALL adherence gaps using ALL data.
#
# Event-time CSVs (step8_survival_time_to_*) are generated locally within
# output/conf_int/gap_*/all_data/ so the comparison PDF uses fully
# self-contained, version-matched data — NOT stale files from output/gap_*/.
#
# Usage:
#   bash code/conf_int/run_all_gaps_all_data.sh [--force]
#
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

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

# Single source of truth for the persistence grids (item 7). This runner drives the
# all-data cohort on the 8 reported thresholds; the 548-day supplementary
# threshold is produced by rerun_conf_int_clean_full.sh. See README.md.
# shellcheck source=../gap_grids.sh
source "$ROOT_DIR/code/gap_grids.sh"
GAPS=("${GAPS_PRIMARY[@]}")
CDIR="$ROOT_DIR/code"
CICODE="$ROOT_DIR/code/conf_int/gap_120"   # scripts work for any gap
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

run_step() {
  local name="$1"; shift
  local outcheck="$1"; shift
  local log="$1"; shift
  local cmd=("$PY" "$@")

  if should_skip "$outcheck" "$name"; then
    return 0
  fi

  echo "[$(_ts)] START ${name}"
  mkdir -p "$(dirname "$log")" "$outcheck"
  rm -f "$(_marker "$outcheck")"

  local rc=0
  ( cd "$ROOT_DIR" && "${cmd[@]}" ) >> "$log" 2>&1 || rc=$?
  if [[ $rc -ne 0 ]]; then
    fail_step "$name" "exit ${rc}; see $log"
    return 0
  fi
  mark_complete "$outcheck"
  echo "[$(_ts)] ${name} OK"
  return 0
}

run_gap() {
  local GAP="$1"
  local OUT="$ROOT_DIR/output/conf_int/gap_${GAP}/all_data"
  local LOG_DIR="$OUT/run_logs"
  mkdir -p "$LOG_DIR"

  local WEIGHT_CSV="$ROOT_DIR/output/step1_prepare_analysis_dataset/analysis_ready_gap${GAP}.csv"
  local A1C_CSV="$ROOT_DIR/output/step1_prepare_analysis_dataset_a1c/analysis_ready_a1c_gap${GAP}.csv"
  local WEIGHT_CONFIG="$ROOT_DIR/output/gap_${GAP}/step2_select_spline_df/model_config.json"
  local A1C_CONFIG="$ROOT_DIR/output/gap_${GAP}/step2_select_spline_df_a1c/model_config_a1c.json"

  # Event CSVs are generated locally — NOT pulled from output/gap_*/
  local WEIGHT_EVENTS="$OUT/step8_survival_time_to_weight_loss/step8_weight_time_to_threshold_events.csv"
  local A1C_EVENTS="$OUT/step8_survival_time_to_a1c_drop/step8_a1c_time_to_threshold_events.csv"

  echo ""
  echo "========================================"
  echo "[$(_ts)] === GAP=${GAP} START ==="
  echo "========================================"

  run_step "step4_predictive_weight" \
    "$OUT/step4_predictive_plots" \
    "$LOG_DIR/step4_predictive_weight.log" \
    "$CICODE/step4_predictive_plots.py" \
    --input-csv "$WEIGHT_CSV" \
    --config-json "$WEIGHT_CONFIG" \
    --outdir "$OUT/step4_predictive_plots" \
    --adherence-gap-days "$GAP"

  run_step "step4_predictive_a1c" \
    "$OUT/step4_predictive_plots_a1c" \
    "$LOG_DIR/step4_predictive_a1c.log" \
    "$CICODE/step4_predictive_plots_a1c.py" \
    --input-csv "$A1C_CSV" \
    --config-json "$A1C_CONFIG" \
    --outdir "$OUT/step4_predictive_plots_a1c" \
    --adherence-gap-days "$GAP"

  run_step "step4_observed_summaries" \
    "$OUT/step4_observed_summary_plots" \
    "$LOG_DIR/step4_observed_summaries.log" \
    "$CICODE/step4_observed_summary_plots.py" \
    --weight-csv "$WEIGHT_CSV" \
    --a1c-csv "$A1C_CSV" \
    --outdir "$OUT/step4_observed_summary_plots" \
    --adherence-gap-days "$GAP" \
    --max-days "$MAX_DAYS" \
    --bin-width 90

  run_step "step6_stratified_weight" \
    "$OUT/step6_stratified_by_covariates_weight" \
    "$LOG_DIR/step6_stratified_weight.log" \
    "$CICODE/step6_stratified_by_covariates_weight.py" \
    --input-csv "$WEIGHT_CSV" \
    --config-json "$WEIGHT_CONFIG" \
    --outdir "$OUT/step6_stratified_by_covariates_weight" \
    --adherence-gap-days "$GAP"

  run_step "step6_stratified_a1c" \
    "$OUT/step6_stratified_by_covariates_a1c" \
    "$LOG_DIR/step6_stratified_a1c.log" \
    "$CICODE/step6_stratified_by_covariates_a1c.py" \
    --input-csv "$A1C_CSV" \
    --config-json "$A1C_CONFIG" \
    --outdir "$OUT/step6_stratified_by_covariates_a1c" \
    --adherence-gap-days "$GAP"

  run_step "step6cc_3way_weight" \
    "$OUT/step6cc_3way_by_age_sex_weight" \
    "$LOG_DIR/step6cc_3way_weight.log" \
    "$CICODE/step6cc_3waystrat_covariates_weight.py" \
    --input-csv "$WEIGHT_CSV" \
    --outdir "$OUT/step6cc_3way_by_age_sex_weight" \
    --adherence-gap-days "$GAP" \
    --max-days "$MAX_DAYS"

  run_step "step6cc_3way_a1c" \
    "$OUT/step6cc_3way_by_age_sex_a1c" \
    "$LOG_DIR/step6cc_3way_a1c.log" \
    "$CICODE/step6cc_3waystrat_covariates_a1c.py" \
    --input-csv "$A1C_CSV" \
    --outdir "$OUT/step6cc_3way_by_age_sex_a1c" \
    --adherence-gap-days "$GAP" \
    --max-days "$MAX_DAYS"

  # ── Step 8: Generate event CSVs locally ────────────────────────────
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

  # ── Step 8: Survival plots + Cox (uses locally-generated events) ────
  run_step "step8_survival_plots" \
    "$OUT/step8_survival_plots_and_cox" \
    "$LOG_DIR/step8_survival.log" \
    "$CICODE/step8_survival_plots_and_cox.py" \
    --weight-events-csv "$WEIGHT_EVENTS" \
    --a1c-events-csv "$A1C_EVENTS" \
    --analysis-a1c-csv "$A1C_CSV" \
    --analysis-weight-csv "$WEIGHT_CSV" \
    --outdir-base "$OUT/step8_survival_plots_and_cox" \
    --figdir-base "$OUT/step8_survival_plots_and_cox/plots" \
    --adherence-gap-days "$GAP"

  echo "[$(_ts)] === GAP=${GAP} DONE. Logs in $LOG_DIR ==="
}

for G in "${GAPS[@]}"; do
  run_gap "$G"
done

echo ""
if [[ ${#FAILED_STEPS[@]} -eq 0 ]]; then
  echo "[$(_ts)] All gaps complete — all steps OK."
else
  echo "[$(_ts)] Finished with ${#FAILED_STEPS[@]} failed step(s):"
  for s in "${FAILED_STEPS[@]}"; do echo "    FAILED: $s"; done
  exit 1
fi
