#!/usr/bin/env bash
# Run add_unstructured analyses for all adherence gap cohorts.
#
# For each gap, this:
#   1. Applies adherence censoring to the prepared assessment data
#   2. Runs ITS, GEE trajectory, CFB, and elevated subgroup analyses
#   3. Writes output to output/conf_int/gap_<N>/add_unstructured/
#
# Usage:
#   bash code/add_unstructured/run_all_gaps.sh [--force]
#
# --force : re-run all gaps even if output already exists

# ── PART II v2 (code-review item 7) ───────────────────────────────────────────
# The gap grid is no longer declared here; all four runners source gap_grids.sh.
# The note-derived analyses run on the 9-threshold grid including 548, matching
# the phase in rerun_conf_int_clean_full.sh that produced the submitted
# supplementary figure sets. See gap_grids.sh and README.md.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

# shellcheck source=../gap_grids.sh
source "$ROOT_DIR/code/gap_grids.sh"
GAPS=("${GAPS_WITH_548[@]}")

FORCE_FLAG=""
for arg in "$@"; do [[ "$arg" == "--force" ]] && FORCE_FLAG="--force"; done
PY=${PYTHON_BIN:-python3}
SCRIPT="$ROOT_DIR/code/add_unstructured/run_for_gap.py"
export MPLBACKEND=Agg

_ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[$(_ts)] Starting add_unstructured gap analyses"
echo "  Gaps:  ${GAPS[*]}"
echo "  Root:  $ROOT_DIR"
echo ""

FAILED=()

for gap in "${GAPS[@]}"; do
    echo "[$(_ts)] ──────────────────────────────────────"
    echo "[$(_ts)] Gap ${gap} days"
    echo "[$(_ts)] ──────────────────────────────────────"

    if $PY "$SCRIPT" --gap "$gap" $FORCE_FLAG; then
        echo "[$(_ts)] Gap ${gap}: SUCCESS"
    else
        echo "[$(_ts)] Gap ${gap}: FAILED"
        FAILED+=("$gap")
    fi
    echo ""
done

echo "[$(_ts)] ======================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "[$(_ts)] All gaps completed successfully."
else
    echo "[$(_ts)] FAILED gaps: ${FAILED[*]}"
    exit 1
fi
