#!/bin/bash
set -euo pipefail

# Submit hierarchical per-run DCE coreg with direct antsRegistration Affine+Mattes:
#
#   prep (single job per run)
#   for each DCE nifti:
#       local_reg array: one task per volume in that nifti
#       local_mean single job
#       mean_to_global single job
#       final_apply array: one task per volume in that nifti
#   finalize (single job per run, after all final_apply arrays)
#
# Usage:
#   submit_dce_hierarchical_coreg_affine_all_runs.bash \
#       /path/to/submit_slurm_cluster_job.bash \
#       /path/to/dce_hierarchical_coreg_affine.py \
#       /mnt/newStor/paros/paros_MRI/DennisTurner/all_niis \
#       "z*" \
#       "12000M" \
#       "0.05" \
#       "100x100x100x20" \
#       "1e-7" \
#       "15" \
#       "8x4x2x1" \
#       "3x2x1x0vox"
#
# Args:
#   1) submit_script
#   2) pipeline_py
#   3) all_niis_dir
#   4) run_glob
#   5) mem
#   6) affine_step
#   7) conv_iters
#   8) conv_thresh
#   9) conv_window
#  10) shrink_factors
#  11) smoothing_sigmas

submit_script="$1"
pipeline_py="$2"
all_niis_dir="$3"
run_glob="$4"
mem="$5"
affine_step="$6"
conv_iters="$7"
conv_thresh="$8"
conv_window="$9"
shift 9
shrink_factors="$1"
smoothing_sigmas="$2"

if [[ ! -x "${submit_script}" ]]; then
  echo "ERROR: submit script not executable: ${submit_script}" >&2
  exit 1
fi
if [[ ! -f "${pipeline_py}" ]]; then
  echo "ERROR: pipeline not found: ${pipeline_py}" >&2
  exit 1
fi
if [[ ! -d "${all_niis_dir}" ]]; then
  echo "ERROR: all_niis dir not found: ${all_niis_dir}" >&2
  exit 1
fi

runs=( ${all_niis_dir}/${run_glob} )
if [[ ${#runs[@]} -eq 0 ]]; then
  echo "ERROR: no runs matched ${all_niis_dir}/${run_glob}" >&2
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
  work_dir="${out_dir}/work"
  mkdir -p "${sbatch_dir}" "${work_dir}"

  final_mean="${out_dir}/${runno}_meanDCE_coregistered_allvols.nii.gz"
  t2_out="${out_dir}/${runno}_T2.nii.gz"

  if [[ -f "${final_mean}" && -f "${t2_out}" ]]; then
    echo "SKIP done-ish: ${bn}"
    skipped=$((skipped+1))
    continue
  fi

  echo
  echo "=== SUBMIT ${bn} (runno=${runno}) ==="

  common_args="--affine_step ${affine_step} --conv_iters ${conv_iters} --conv_thresh ${conv_thresh} --conv_window ${conv_window} --shrink_factors ${shrink_factors} --smoothing_sigmas ${smoothing_sigmas} --verbose"

  prep_name="DCEprep_${runno}"
  prep_cmd="python3 ${pipeline_py} prep --run_dir ${run_dir} ${common_args}"
  prep_jobid="$("${submit_script}" "${sbatch_dir}" "${prep_name}" "${mem}" "0" "${prep_cmd}" | tail -1 | cut -d ';' -f1 | awk '{print $4}')"
  echo "prep jobid: ${prep_jobid}"

  final_apply_jobids=()

  dce_files=( "${run_dir}/${runno}_DCE_baseline.nii.gz" "${run_dir}/${runno}_DCE_block1.nii.gz" "${run_dir}/${runno}_DCE_block2.nii.gz" )

  for dce_nifti in "${dce_files[@]}"; do
    [[ -f "${dce_nifti}" ]] || continue

    dce_base="$(basename "${dce_nifti}")"
    dce_tag="${dce_base%.nii.gz}"

    n_tasks="$(python3 - <<PY
import nibabel as nib
img = nib.load(r"${dce_nifti}")
shape = img.shape
if len(shape) != 4:
    raise SystemExit("Expected 4D")
print(shape[3])
PY
)"
    if [[ -z "${n_tasks}" ]]; then
      echo "ERROR: could not determine n_tasks for ${dce_nifti}" >&2
      exit 1
    fi
    array_max=$((n_tasks - 1))

    echo "  ${dce_base}: ${n_tasks} volumes"

    local_array_script="${sbatch_dir}/${dce_tag}_local_reg_array.bash"
    cat > "${local_array_script}" <<EOF
#!/bin/bash
#SBATCH --job-name=LocReg_${runno}
#SBATCH --partition=normal
#SBATCH --mem=${mem}
#SBATCH --output=${sbatch_dir}/slurm-%A_%a.out
#SBATCH --error=${sbatch_dir}/slurm-%A_%a.out
#SBATCH --array=0-${array_max}
#SBATCH --dependency=afterok:${prep_jobid}

set -euo pipefail

python3 "${pipeline_py}" local_reg \
  --run_dir "${run_dir}" \
  --dce_nifti "${dce_nifti}" \
  --affine_step "${affine_step}" \
  --conv_iters "${conv_iters}" \
  --conv_thresh "${conv_thresh}" \
  --conv_window "${conv_window}" \
  --shrink_factors "${shrink_factors}" \
  --smoothing_sigmas "${smoothing_sigmas}" \
  --verbose
EOF
    chmod +x "${local_array_script}"
    local_array_submit="$(sbatch "${local_array_script}")"
    local_array_jobid="$(echo "${local_array_submit}" | awk '{print $4}')"
    echo "    local_reg array jobid: ${local_array_jobid}"

    local_mean_name="LocMean_${runno}_$(echo "${dce_tag}" | tr '/' '_')"
    local_mean_cmd="python3 ${pipeline_py} local_mean --run_dir ${run_dir} --dce_nifti ${dce_nifti} ${common_args}"
    local_mean_jobid="$("${submit_script}" "${sbatch_dir}" "${local_mean_name}" "${mem}" "afterok:${local_array_jobid}" "${local_mean_cmd}" | tail -1 | cut -d ';' -f1 | awk '{print $4}')"
    echo "    local_mean jobid: ${local_mean_jobid}"

    mean_to_global_name="Mean2Glob_${runno}_$(echo "${dce_tag}" | tr '/' '_')"
    mean_to_global_cmd="python3 ${pipeline_py} mean_to_global --run_dir ${run_dir} --dce_nifti ${dce_nifti} ${common_args}"
    mean_to_global_jobid="$("${submit_script}" "${sbatch_dir}" "${mean_to_global_name}" "${mem}" "afterok:${local_mean_jobid}" "${mean_to_global_cmd}" | tail -1 | cut -d ';' -f1 | awk '{print $4}')"
    echo "    mean_to_global jobid: ${mean_to_global_jobid}"

    final_array_script="${sbatch_dir}/${dce_tag}_final_apply_array.bash"
    cat > "${final_array_script}" <<EOF
#!/bin/bash
#SBATCH --job-name=FinalApply_${runno}
#SBATCH --partition=normal
#SBATCH --mem=${mem}
#SBATCH --output=${sbatch_dir}/slurm-%A_%a.out
#SBATCH --error=${sbatch_dir}/slurm-%A_%a.out
#SBATCH --array=0-${array_max}
#SBATCH --dependency=afterok:${mean_to_global_jobid}

set -euo pipefail

python3 "${pipeline_py}" final_apply \
  --run_dir "${run_dir}" \
  --dce_nifti "${dce_nifti}" \
  --verbose
EOF
    chmod +x "${final_array_script}"
    final_array_submit="$(sbatch "${final_array_script}")"
    final_array_jobid="$(echo "${final_array_submit}" | awk '{print $4}')"
    echo "    final_apply array jobid: ${final_array_jobid}"

    final_apply_jobids+=( "${final_array_jobid}" )
  done

  if [[ ${#final_apply_jobids[@]} -eq 0 ]]; then
    echo "No DCE files found for ${bn}; skipping finalize" >&2
    continue
  fi

  dep_list="$(IFS=:; echo "${final_apply_jobids[*]}")"
  finalize_dep="afterok:${dep_list}"

  finalize_name="DCEfinal_${runno}"
  finalize_cmd="python3 ${pipeline_py} finalize --run_dir ${run_dir} ${common_args}"
  finalize_jobid="$("${submit_script}" "${sbatch_dir}" "${finalize_name}" "${mem}" "${finalize_dep}" "${finalize_cmd}" | tail -1 | cut -d ';' -f1 | awk '{print $4}')"
  echo "finalize jobid: ${finalize_jobid}"

  submitted=$((submitted+1))
done

echo
echo "Submitted runs: ${submitted}"
echo "Skipped runs:   ${skipped}"