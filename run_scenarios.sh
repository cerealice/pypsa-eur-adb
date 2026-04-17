#!/bin/bash
# run_scenarios.sh
#
# Submits each scenario as a chained bsub job so they run sequentially —
# each job starts only after the previous one finishes successfully.
#
# Usage:
#   bash run_scenarios.sh
#   bash run_scenarios.sh policy_reg_deindustrial policy_reg_maintain   # run a subset

set -euo pipefail

# ---------------------------------------------------------------------------
# Scenarios to run (edit this list as needed)
# ---------------------------------------------------------------------------
DEFAULT_SCENARIOS=(
    "base_reg_maintain"
    "policy_reg_deindustrial"
    "policy_reg_maintain"
    "policy_reg_regain"
)

# Accept a custom list from command-line arguments, otherwise use the default
if [ $# -gt 0 ]; then
    SCENARIOS=("$@")
else
    SCENARIOS=("${DEFAULT_SCENARIOS[@]}")
fi

# ---------------------------------------------------------------------------
# bsub options (same as your usual command)
# ---------------------------------------------------------------------------
QUEUE="p_gams"
MEMORY="256G"
NCORES=72
PROJECT="0607"
BASE_CONFIG="config/industry_config.yaml"
TMP_CONFIG_DIR="config/run_configs"

mkdir -p "$TMP_CONFIG_DIR"

prev_jobid=""

for scenario in "${SCENARIOS[@]}"; do

    tmp_config="${TMP_CONFIG_DIR}/${scenario}.yaml"

    # Create a copy of the base config with only this scenario active
    python3 - <<EOF
import yaml, sys

with open("${BASE_CONFIG}") as f:
    cfg = yaml.safe_load(f)

cfg["run"]["name"] = ["${scenario}"]

with open("${tmp_config}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print(f"  Config written: ${tmp_config}")
EOF

    SNAKEMAKE_CMD="snakemake -call all --configfile=${tmp_config} --rerun-incomplete"

    if [ -z "$prev_jobid" ]; then
        # First job — no dependency
        submit_output=$(bsub \
            -q "$QUEUE" \
            -M "$MEMORY" \
            -n "$NCORES" \
            -x \
            -P "$PROJECT" \
            -J "pypsa_${scenario}" \
            $SNAKEMAKE_CMD 2>&1)
    else
        # Subsequent jobs — wait for the previous job to finish successfully
        submit_output=$(bsub \
            -q "$QUEUE" \
            -M "$MEMORY" \
            -n "$NCORES" \
            -x \
            -P "$PROJECT" \
            -J "pypsa_${scenario}" \
            -w "done(${prev_jobid})" \
            $SNAKEMAKE_CMD 2>&1)
    fi

    # Extract the numeric job ID from bsub output: "Job <12345> is submitted..."
    jobid=$(echo "$submit_output" | grep -oP '(?<=Job <)\d+(?=>)')

    if [ -z "$jobid" ]; then
        echo "ERROR: failed to submit scenario '${scenario}'. bsub output:"
        echo "$submit_output"
        exit 1
    fi

    echo "Queued '${scenario}' → job ${jobid}$([ -n "$prev_jobid" ] && echo " (waiting for ${prev_jobid})" || true)"
    prev_jobid="$jobid"

done

echo ""
echo "All ${#SCENARIOS[@]} scenarios queued. Check status with: bjobs"
