#!/usr/bin/env bash
set -euo pipefail

# submit_coregistered_roi_plots_autodiscover.sh
#
# Autodiscovers z${runno} folders, finds ROI masks there, and submits one Slurm
# job per runno to run plot_roi_intensity.py using images from:
#
#   all_niis/${runno}_coregistered
#
# while sourcing ROI mask definitions from:
#
#   all_niis/z${runno}
#
# Expected mask naming pattern:
#   ${runno}_${REGION}_roi_mask_xyz_X_Y_Z_rR.nii.gz
#
# where REGION is one of:
#   CM, CSF, Hc
#
# Example:
#   bash submit_coregistered_roi_plots_autodiscover.sh
#
# Optional:
#   bash submit_coregistered_roi_plots_autodiscover.sh /custom/base/all_niis
#
# Assumes environment variables:
#   MRI
#   WORK
#
# Plot script used:
#   $WORK/DCE_dev/Bruker_radial_recon/plot_roi_intensity.py
#
# Submitted job behavior:
#   - one job per runno
#   - each job runs up to 3 regions sequentially: CM, CSF, Hc
#
# Output structure:
#   all_niis/${runno}_coregistered/sbatch_roi_plots/
#       submit_*.sbatch
#       logs/
#       helper_run_*.sh
#
# Default plotting command mirrors your recent usage:
#   --img2-baseline ${runno}_DCE_baseline.nii.gz
#   --norm-method affine2boundline
#   --affine-source roi
#   --title-tag "${runno} ${region}"
#   --title-mode compact
#   --time-unit min
#   --affine-solver robust
#   --std-max-ratio 1.7
#   --a-positive

###############################################################################
# User-tunable defaults
###############################################################################

: "${MRI:?ERROR: MRI environment variable is not set}"
: "${WORK:?ERROR: WORK environment variable is not set}"

ALL_NIIS_BASE_DEFAULT="$MRI/DennisTurner/all_niis"
ALL_NIIS_BASE="${1:-$ALL_NIIS_BASE_DEFAULT}"

PLOT_SCRIPT="$WORK/DCE_dev/Bruker_radial_recon/plot_roi_intensity.py"

# Slurm resources
SBATCH_PARTITION="${SBATCH_PARTITION:-}"
SBATCH_TIME="${SBATCH_TIME:-01:00:00}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-1}"
SBATCH_MEM="${SBATCH_MEM:-8G}"

# Whether to require all three masks. If false, submit/run whatever exists.
REQUIRE_ALL_THREE_MASKS="${REQUIRE_ALL_THREE_MASKS:-0}"

# Regions to look for
REGIONS=(CM CSF Hc)

###############################################################################
# Sanity checks
###############################################################################

if [[ ! -d "$ALL_NIIS_BASE" ]]; then
    echo "ERROR: all_niis base directory not found: $ALL_NIIS_BASE" >&2
    exit 1
fi

if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "ERROR: plot script not found: $PLOT_SCRIPT" >&2
    exit 1
fi

###############################################################################
# Helpers
###############################################################################

log() {
    echo "[$(date '+%F %T')] $*"
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

parse_mask_xyzr() {
    local mask_path="$1"
    local base
    base="$(basename "$mask_path")"

    # Matches:
    #   ..._roi_mask_xyz_57_38_49_r2.5.nii.gz
    #   ..._roi_mask_xyz_57_38_49_r3.nii.gz
    if [[ "$base" =~ xyz_([0-9]+)_([0-9]+)_([0-9]+)_r([0-9]+([.][0-9]+)?) ]]; then
        echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]} ${BASH_REMATCH[3]} ${BASH_REMATCH[4]}"
        return 0
    fi

    return 1
}

find_first_mask_for_region() {
    local zdir="$1"
    local runno="$2"
    local region="$3"
    local -a matches

    shopt -s nullglob
    matches=( "$zdir/${runno}_${region}_roi_mask_"*.nii.gz )
    shopt -u nullglob

    if [[ ${#matches[@]} -eq 0 ]]; then
        return 1
    fi

    printf '%s\n' "${matches[0]}"
    return 0
}

submit_one_runno() {
    local runno="$1"
    local zdir="$2"
    local coreg_dir="$3"

    local img1="$coreg_dir/${runno}_DCE_block1.nii.gz"
    local img2="$coreg_dir/${runno}_DCE_block2.nii.gz"
    local baseline="$coreg_dir/${runno}_DCE_baseline.nii.gz"

    if [[ ! -f "$img1" ]]; then
        log "SKIP $runno : missing $img1"
        return 0
    fi
    if [[ ! -f "$img2" ]]; then
        log "SKIP $runno : missing $img2"
        return 0
    fi
    if [[ ! -f "$baseline" ]]; then
        log "SKIP $runno : missing $baseline"
        return 0
    fi

    local sbatch_dir="$coreg_dir/sbatch_roi_plots"
    local log_dir="$sbatch_dir/logs"
    mkdir -p "$sbatch_dir" "$log_dir"

    local helper_script="$sbatch_dir/helper_run_${runno}_roi_plots.sh"
    local submit_script="$sbatch_dir/submit_${runno}_roi_plots.sbatch"

    # Gather region/mask info now so the helper script is fully explicit.
    local found_count=0
    local region
    local mask
    local parsed
    local x y z r

    local helper_body=""
    for region in "${REGIONS[@]}"; do
        if mask="$(find_first_mask_for_region "$zdir" "$runno" "$region")"; then
            if ! parsed="$(parse_mask_xyzr "$mask")"; then
                log "SKIP region $region for $runno : could not parse xyz/r from mask $(basename "$mask")"
                continue
            fi
            read -r x y z r <<< "$parsed"

            helper_body+=$'\n'
            helper_body+="echo \"===================================================================\""$'\n'
            helper_body+="echo \"RUNNO : $runno\""$'\n'
            helper_body+="echo \"REGION: $region\""$'\n'
            helper_body+="echo \"MASK  : $mask\""$'\n'
            helper_body+="echo \"CENTER: $x $y $z\""$'\n'
            helper_body+="echo \"RADIUS: $r\""$'\n'
            helper_body+="echo \"===================================================================\""$'\n'
            helper_body+="python \"$PLOT_SCRIPT\" \\"$'\n'
            helper_body+="    \"$img1\" \\"$'\n'
            helper_body+="    \"$img2\" \\"$'\n'
            helper_body+="    --img2-baseline \"$baseline\" \\"$'\n'
            helper_body+="    --radius \"$r\" \\"$'\n'
            helper_body+="    --norm-method affine2boundline \\"$'\n'
            helper_body+="    --affine-source roi \\"$'\n'
            helper_body+="    --title-tag \"$runno $region\" \\"$'\n'
            helper_body+="    --title-mode compact \\"$'\n'
            helper_body+="    --time-unit min \\"$'\n'
            helper_body+="    --center \"$x\" \"$y\" \"$z\" \\"$'\n'
            helper_body+="    --affine-solver robust \\"$'\n'
            helper_body+="    --std-max-ratio 1.7 \\"$'\n'
            helper_body+="    --a-positive"$'\n'

            ((found_count+=1))
        else
            log "INFO $runno : no mask found for region $region"
        fi
    done

    if [[ "$REQUIRE_ALL_THREE_MASKS" == "1" && "$found_count" -ne 3 ]]; then
        log "SKIP $runno : REQUIRE_ALL_THREE_MASKS=1 but found $found_count/3 masks"
        return 0
    fi

    if [[ "$found_count" -eq 0 ]]; then
        log "SKIP $runno : no usable masks found"
        return 0
    fi

    cat > "$helper_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

echo "[\$(date '+%F %T')] Starting ROI plot helper for $runno"
echo "[\$(date '+%F %T')] Host: \$(hostname)"
echo "[\$(date '+%F %T')] PWD : \$(pwd)"

$helper_body

echo "[\$(date '+%F %T')] Finished ROI plot helper for $runno"
EOF
    chmod +x "$helper_script"

    cat > "$submit_script" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=roi_${runno}
#SBATCH --output=$log_dir/%x_%j.out
#SBATCH --error=$log_dir/%x_%j.err
#SBATCH --time=$SBATCH_TIME
#SBATCH --cpus-per-task=$SBATCH_CPUS_PER_TASK
#SBATCH --mem=$SBATCH_MEM
EOF

    if [[ -n "$SBATCH_PARTITION" ]]; then
        echo "#SBATCH --partition=$SBATCH_PARTITION" >> "$submit_script"
    fi

    cat >> "$submit_script" <<EOF

set -euo pipefail

echo "[\$(date '+%F %T')] Slurm job starting for $runno"
echo "[\$(date '+%F %T')] Node: \$(hostname)"
echo "[\$(date '+%F %T')] JobID: \$SLURM_JOB_ID"

"$helper_script"

echo "[\$(date '+%F %T')] Slurm job finished for $runno"
EOF

    chmod +x "$submit_script"

    local submit_out
    submit_out="$(sbatch "$submit_script")"
    log "SUBMITTED $runno : $submit_out"
}

###############################################################################
# Main autodiscovery loop
###############################################################################

log "Starting autodiscovery under: $ALL_NIIS_BASE"

shopt -s nullglob
zdirs=( "$ALL_NIIS_BASE"/z* )
shopt -u nullglob

if [[ ${#zdirs[@]} -eq 0 ]]; then
    fail "No z* folders found under $ALL_NIIS_BASE"
fi

submitted=0
skipped=0

for zdir in "${zdirs[@]}"; do
    [[ -d "$zdir" ]] || continue

    zbase="$(basename "$zdir")"
    if [[ ! "$zbase" =~ ^z(.+)$ ]]; then
        log "SKIP non-run folder: $zdir"
        ((skipped+=1))
        continue
    fi

    runno="${BASH_REMATCH[1]}"
    coreg_dir="$ALL_NIIS_BASE/${runno}_coregistered"

    if [[ ! -d "$coreg_dir" ]]; then
        log "SKIP $runno : missing coregistered dir $coreg_dir"
        ((skipped+=1))
        continue
    fi

    if submit_one_runno "$runno" "$zdir" "$coreg_dir"; then
        ((submitted+=1))
    else
        ((skipped+=1))
    fi
done

log "Done. submit attempts processed: $submitted"