#!/bin/bash
set -euo pipefail

# Submit one job per run folder (all_niis/z*) using submit_slurm_cluster_job.bash,
# writing sbatch scripts + slurm outs into all_niis/${runno}_coregistered/sbatch/.
#
# Usage:
#   submit_dce_coreg_all_runs.bash \
#     /path/to/submit_slurm_cluster_job.bash \
#     /path/to/dce_coregister_run.py \
#     /mnt/newStor/paros/paros_MRI/DennisTurner/all_niis \
#     "z*" \
#     "12000M" \
#     "r" \
#     "r"
#
# Args:
#   1) submit_script          Path to submit_slurm_cluster_job.bash
#   2) python_pipeline        Path to dce_coregistration_pipeline.py
#   3) all_niis_dir           Folder containing run dirs (z*)
#   4) run_glob               Glob for run dirs under all_niis_dir (e.g. "z*")
#   5) memory                 Memory string (e.g. 12000M). Use 0 to default.
#   6) t2_to_mean_xfm         r|a|s
#   7) dce_to_t2_xfm          r|a|s

submit_script="$1"
python_pipeline="$2"
all_niis_dir="$3"
run_glob="$4"
memory="$5"
t2_to_mean_xfm="$6"
dce_to_t2_xfm="$7"

if [[ ! -x "${submit_script}" ]]; then
  echo "ERROR: submit_script not executable: ${submit_script}" >&2
  exit 1
fi
if [[ ! -f "${python_pipeline}" ]]; then
  echo "ERROR: python pipeline not found: ${python_pipeline}" >&2
  exit 1
fi
if [[ ! -d "${all_niis_dir}" ]]; then
  echo "ERROR: all_niis_dir not found: ${all_niis_dir}" >&2
  exit 1
fi

runs=( ${all_niis_dir}/${run_glob} )
if [[ ${#runs[@]} -eq 0 ]]; then
  echo "No runs matched: ${all_niis_dir}/${run_glob}" >&2
  exit 1
fi

submitted=0
skipped=0

for run_dir in "${runs[@]}"; do
  [[ -d "${run_dir}" ]] || continue

  bn="$(basename "${run_dir}")"
  if [[ "${bn}" == z* ]]; then
    runno="${bn#z}"
  else
    runno="${bn}"
  fi

  out_dir="${all_niis_dir}/${runno}_coregistered"
  sbatch_dir="${out_dir}/sbatch"
  mkdir -p "${sbatch_dir}"
  mkdir -p "${out_dir}/work"

  # Skip if clearly done
  mean_i="${out_dir}/${runno}_meanDCE_initial_allvols.nii.gz"
  mean_c="${out_dir}/${runno}_meanDCE_coregistered_allvols.nii.gz"
  t2_out_glob="${out_dir}/${runno}_T2.nii.gz"

  if [[ -f "${mean_i}" && -f "${mean_c}" && -f "${t2_out_glob}" ]]; then
    echo "SKIP (already done-ish): ${bn} -> ${runno}_coregistered"
    skipped=$((skipped+1))
    continue
  fi

  job_name="DCEcoreg_${runno}"

  # If you need environment activation on compute nodes, prepend it here.
  # Example:
  # cmd="source /path/to/conda.sh; conda activate myenv; python3 ..."
  cmd="python3 ${python_pipeline} \
    --run_dir ${run_dir} \
    --ants_transform_t2_to_mean ${t2_to_mean_xfm} \
    --ants_transform_dce_to_t2 ${dce_to_t2_xfm} \
    --verbose"

  "${submit_script}" "${sbatch_dir}" "${job_name}" "${memory}" "0" "${cmd}"
  submitted=$((submitted+1))
done

echo
echo "Submitted: ${submitted}"
echo "Skipped:   ${skipped}"