Requirements:
Conda (Miniconda/Mambaforge recommended)

One-time setup from the repo root (where environment.yml lives):
```bash
bash ./setup_env.sh
```

To activate env:
```bash
conda activate ./.envs/mri-recon
```
To verify environment has been setup successfully:
```bash
bash ./verify_environment.sh
```

Notes

Modality: 
``` --modality dwi ```
uses log1p/expm1; 
``` --modality fmri ```
uses Fisher z (arctanh/tanh).
