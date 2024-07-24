#!/bin/bash
#SBATCH --account=msaraclar
#SBATCH --partition=palamut-cuda
#SBATCH --job-name=sign2vec_pretrain
#SBATCH --output=%A.out
#SBATCH --error=%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH	--time=16:00:00
###SBATCH --mal-user= your_email_address
###SBATCH --mail-type=BEGIN,END,FAIL
###SBATCH --mail-type=ALL


### Load modules
module purge
hostname
nvidia-smi

# Load conda environment
eval "$(/truba/home/$USER/miniconda3/bin/conda shell.bash hook)"
conda activate pyt-env
which python

# Change the directory to the sign2vec repository
cd /truba/home/msaraclar/sign2vec

# Change the directory to the YASL poses
mkdir -p /localscratch/msaraclar/data/
mkdir -p /localscratch/msaraclar/sign2vec/
cp -r /truba/home/msaraclar/data/YASL /localscratch/msaraclar/data/

# Set environment variables
export HF_TOKEN="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"
export WANDB_API_KEY="45373c78d8dfa3b3cfd4694785b7f1f0ffbcf570"

# Run the get pointer script
python3 sign2vec/utils/generate_pointer.py --pose_dir /localscratch/msaraclar/data/ \
                                           --output_dir /localscratch/msaraclar/data/

# Run the pretraining script
TORCHDYNAMO_VERBOSE=1 accelerate launch pretraining/run_sign2vec_pretraining.py \
                                                    --config_name="pretraining/training_configuration/truba/youtube_asl_mc_sc.yaml" 

# Remove the YASL poses
rm -rf /localscratch/msaraclar/data
rm -rf /localscratch/msaraclar/sign2vec

# Save the job info
scontrol show job $SLURM_JOB_ID >> ${SLURM_JOB_ID}.info
