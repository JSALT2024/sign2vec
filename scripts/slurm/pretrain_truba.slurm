#!/bin/bash
#SBATCH --account=msaraclar
#SBATCH --partition=palamut-cuda
#SBATCH --job-name=sign2vec_pretrain
#SBATCH --output=slurm/%A.out
#SBATCH --error=slurm/%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH	--time=24:00:00
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

if [ -d "/localscratch/msaraclar/data" ]; then
    echo "Directory exists. Cleaning the directory."
    rm -rf /localscratch/msaraclar/data
else
    echo "Directory does not exist. "
fi

echo "Copying the data to the local scratch. Creating the directory."
# Change the directory to the YASL poses
mkdir -p /localscratch/msaraclar/data
mkdir -p /localscratch/msaraclar/sign2vec
cp -r /truba/home/msaraclar/data/YASL /localscratch/msaraclar/data/

ls /localscratch/msaraclar/data/

echo "Finish copying to the local scratch."
date
    
# Set environment variables
export HF_TOKEN="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"
export WANDB_API_KEY="45373c78d8dfa3b3cfd4694785b7f1f0ffbcf570"

echo "Start generating pointer file."

# Run the get pointer script
python3 sign2vec/utils/generate_pointer.py --pose_dir /localscratch/msaraclar/data/YASL \
                                           --output_path /localscratch/msaraclar/data

echo "Finish generating pointer file."

echo "Start pretraining the model."
date
# Run the pretraining script
TORCHDYNAMO_VERBOSE=1 accelerate launch pretraining/run_sign2vec_pretraining.py \
                                                    --config_name="pretraining/training_configuration/truba/youtube_asl_mc_sc.yaml" 


echo "Finish pretraining the model."
date

echo "Removing the data from the local scratch."
# Remove the YASL poses
rm -rf /localscratch/msaraclar/data
rm -rf /localscratch/msaraclar/sign2vec

echo "Finish removing the data from the local scratch."

date

# Save the job info
scontrol show job $SLURM_JOB_ID >> ${SLURM_JOB_ID}.info
