#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE SOURCE OF TRUTH for the persistence ("adherence gap") grids.
#
# Part II v2 (2026-08-17). Addresses code-review item 7: the four batch runners
# each declared their own gap list and they disagreed — rerun_conf_int_clean_full.sh
# and add_unstructured/run_all_gaps.sh included the 548-day threshold, while
# conf_int/run_all_gaps_all_data.sh and conf_int/run_all_gaps_structured_only.sh
# did not. Every runner now sources this file instead of declaring its own list,
# so the grids cannot drift apart again.
#
#   source "$(dirname "$0")/gap_grids.sh"        # from code/
#   source "$(dirname "$0")/../gap_grids.sh"     # from code/conf_int/ etc.
#
# ── What each grid is, and why they differ ────────────────────────────────────
#
# GAPS_PRIMARY (8)   30 60 90 120 150 180 365 730
#     The eight persistence thresholds reported in the manuscript. The Methods
#     state: "This process was applied separately for each of the eight gap
#     thresholds evaluated: 30, 60, 90, 120, 150, 180, 365, and 730 days.
#     Primary analyses used g = 120 days." This is the grid behind every
#     reported estimate. The supplementary sample-size table also uses the
#     8-threshold layout.
#
# GAPS_WITH_548 (9)  30 60 90 120 150 180 365 548 730
#     The eight above plus a 548-day (~18-month) threshold. 548 is NOT a
#     reported threshold; it appears only in supplementary figure sets and in
#     the sample-size grid, where it was run to show the trajectory at the
#     18-month follow-up boundary. It is retained deliberately, not dropped,
#     because the submitted supplementary figures were produced with it —
#     see the script-to-deliverable map in README.md.
#
# GAPS_STRUCTURED (8)  30 60 90 120 150 180 365 730
#     The structured-only sensitivity cohort. 548 was never run here, in either
#     the submitted or the v2 package, because the structured-only step1 data
#     was built on the 8-threshold grid. Kept as its own name (rather than an
#     alias of GAPS_PRIMARY) so that the distinction stays visible if the
#     structured-only grid is ever extended.
#
# The manuscript numbers were produced by rerun_conf_int_clean_full.sh, whose
# all-data and note-derived phases use GAPS_WITH_548 and whose structured-only
# phase uses GAPS_STRUCTURED.
# ═══════════════════════════════════════════════════════════════════════════════

GAPS_PRIMARY=(30 60 90 120 150 180 365 730)
GAPS_WITH_548=(30 60 90 120 150 180 365 548 730)
GAPS_STRUCTURED=(30 60 90 120 150 180 365 730)

# Primary (prespecified) persistence definition, in days.
GAP_PRIMARY=120

# Maximum follow-up retained by step1, in days. See README.md ("Follow-up
# horizon caps") for how this relates to the 540/548-day caps downstream.
MAX_DAYS=730
