$WITH_CONDA

source /venvs/signllava-venv/bin/activate
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

export PYTHONPATH=$PYTHONPATH:/venvs/signllava-venv/lib/python3.12/site-packages/
cd /scratch/project_465000977/eleznyto/sign2vec/sign2vec
export PYTHONPATH=$PYTHONPATH:$(pwd)/

export PATH=$PATH:/pfs/lustrep2/scratch/project_465000977/eleznyto/venvs/signllava-venv/bin/


export LC_ALL=C
export WANDB_PROJECT="T5_YTASL"
export WANDB_API_KEY="your-sexy-wandb-token"
export WANDB_ENTITY="jsalt2024-slt"
export WANDB_RUN_GROUP="test-run"

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NPROCS

# Set interfaces to be used by RCCL.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

echo "Running training script.."
python sign2vec/train/run_finetuning.py \
	--learning_rate=0.0004 \
	--annotation_file=/pfs/lustrep2/scratch/project_465000977/data/YoutubeASL/features \
	--metadata_file=/scratch/project_465000977/data/YoutubeASL/features_v2/keypoints \
	--dataset_type=yasl \
	--model_id=google-t5/t5-base \
	--max_training_step=20000 \
	--per_device_train_batch_size=32 \
	--per_device_eval_batch_size=2 \
	--gradient_accumulation_steps=4 \
	--lr_scheduler_type=linear \
	--max_sequence_length=250 \
	--max_token_length=256 \
	--logging_steps=20 \
	--eval_steps=50 \
	--project_name=T5_YTASL \
	--model_name=my-sexy-test-run \
	--is_normalized \
	--skip_frames
