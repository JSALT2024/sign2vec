#!/usr/bin/env -S bash -e
#SBATCH --job-name=T5_YTASL_pretrain
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --output="/pfs/lustrep2/scratch/project_465000977/eleznyto/sign2vec/logs/t5/output_%x_%j.txt"
#SBATCH --partition=standard-g
#SBATCH --time=48:00:00
#SBATCH --account=project_465000977

export EBU_USER_PREFIX=/project/project_465000977/EasyBuild
module load CrayEnv

SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif

PROJECT_PATH=/scratch/project_465000977
VENV_PATH=/scratch/project_465000977/eleznyto/venvs
HOME_PATH=/scratch/project_465000977/eleznyto/SignLLMpp
OMNITRACE=/pfs/lustrep3/scratch/project_462000394

export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=50000

srun --cpu-bind=mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000 \
    singularity exec -B $OMNITRACE,$HOME_PATH,${HOME_PATH}/Sign_LLaVA,$PROJECT_PATH --bind /pfs,/scratch,/projappl,/project,/flash,/appl -B $VENV_PATH:/venvs $SIF \
    bash /users/eleznyto/sign2vec/pretrain/train_ytasl_T5_linear.sh

#> /pfs/lustrep2/scratch/project_465000977/profile/logs/output_${SLURM_JOB_NAME}.txt
