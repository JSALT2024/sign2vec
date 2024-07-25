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

export LOCALDIR = "/localscratch/msaraclar/$SLURM_JOB_ID"

if [ -d  $LOCALDIR ]; then
    echo "Directory exists. Cleaning the directory."
    rm -rf $LOCALDIR
else
    echo "Directory does not exist. "
fi

echo "Copying the data to the local scratch. Creating the directory."
# Change the directory to the YASL poses
mkdir -p $LOCALDIR
mkdir -p $LOCALDIR/sign2vec
mkdir -p $LOCALDIR/data
cp -r /truba/home/msaraclar/data/YASL $LOCALDIR/data

ls $LOCALDIR/data

echo "Finish copying to the local scratch."
date
    
# Set environment variables
export HF_TOKEN="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"
export WANDB_API_KEY="45373c78d8dfa3b3cfd4694785b7f1f0ffbcf570"

echo "Start generating pointer file."

# Run the get pointer script
python3 sign2vec/utils/generate_pointer.py --pose_dir $LOCALDIR/data/YASL \
                                           --output_path $LOCALDIR/data

echo "Finish generating pointer file."

echo "Start pretraining the model."
date
# Run the pretraining script
TORCHDYNAMO_VERBOSE=1 accelerate launch pretraining/run_sign2vec_pretraining.py \
                                                    --config_name="pretraining/training_configuration/truba/youtube_asl_mc_sc.yaml" \
                                                    --data_dir="$LOCALDIR/data/YASL" \
                                                    --train_data_path="$LOCALDIR/data/train_dataset.csv" \
                                                    --validation_data_path="$LOCALDIR/data/val_dataset.csv" \
                                                    --output_dir="$LOCALDIR/sign2vec"

echo "Finish pretraining the model."
date

echo "Removing the files from the local scratch."
# Remove the YASL poses
rm -rf $LOCALDIR
echo "Finished removing the files from the local scratch."
date

# Save the job info
scontrol show job $SLURM_JOB_ID >> slurm/${SLURM_JOB_ID}.info
