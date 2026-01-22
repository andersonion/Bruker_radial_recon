#!/usr/bin/env bash
# diff_and_pctchange_4d.sh
# Make (last - first) difference image and % change image from a 4D NIfTI.
#
# Usage:
#   ./diff_and_pctchange_4d.sh in4d.nii.gz [out_prefix]
#
# Outputs:
#   <out_prefix>_first.nii.gz
#   <out_prefix>_last.nii.gz
#   <out_prefix>_diff_last_minus_first.nii.gz
#   <out_prefix>_pctchange_100x.nii.gz          # 100*(last-first)/first
#   <out_prefix>_pctchange_100x_eps.nii.gz      # same but denominator clamped to eps to avoid blowups

set -euo pipefail

in="${1:?need input 4D nifti}"
prefix="${2:-${in%.nii*}}"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

first="$tmpdir/first.nii.gz"
last="$tmpdir/last.nii.gz"
diff="${prefix}_diff_last_minus_first.nii.gz"
pct="${prefix}_pctchange_100x.nii.gz"
pct_eps="${prefix}_pctchange_100x_eps.nii.gz"

# Extract first and last 3D volumes
fslroi "$in" "$first" 0 1
nvols="$(fslnvols "$in")"
last_idx=$((nvols - 1))
fslroi "$in" "$last" "$last_idx" 1

# Save first/last too (handy for QA)
imcp "$first" "${prefix}_first.nii.gz"
imcp "$last"  "${prefix}_last.nii.gz"

# Difference: (last - first)
fslmaths "$last" -sub "$first" "$diff"

# % change: 100*(last-first)/first
fslmaths "$diff" -div "$first" -mul 100 "$pct"

# Safer % change: clamp |first| to eps to reduce infs where baseline ~0
eps="1e-6"
fslmaths "$first" -abs -thr "$eps" "${tmpdir}/den.nii.gz"
fslmaths "$diff" -div "${tmpdir}/den.nii.gz" -mul 100 "$pct_eps"

echo "Wrote:"
echo "  ${prefix}_first.nii.gz"
echo "  ${prefix}_last.nii.gz"
echo "  $diff"
echo "  $pct"
echo "  $pct_eps"
