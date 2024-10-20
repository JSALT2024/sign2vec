# get bash parameter run_name
run_name=$1
cuda_device=$2
export CUDA_VISIBLE_DEVICES=$cuda_device

TORCHDYNAMO_VERBOSE=1 accelerate launch sign2vec/train/run_sign2vec_pretraining.py \
                    --dataset_name="YoutubeASL" \
                    --run_name=$1 \
                    --hub_model_id="sign2vec-base" \
                    --tags sign2vec base yasl-norm single_cue \
                    --datasets "train" "test" \
                    --dataset_path="/ssd2/karahan/YASL/pose"  \
                    --model_config_file="experimental/configs/sign2vec_pretraining_config.yaml" \
                    --output_dir="/ssd2/karahan/YASL/sign2vec-base" \
                    --max_train_steps="20000" \
                    --num_warmup_steps="2000" \
                    --gradient_accumulation_steps="4" \
                    --learning_rate="0.0005" \
                    --weight_decay="0.001" \
                    --max_duration_in_seconds="20.0" \
                    --min_duration_in_seconds="2.0" \
                    --logging_steps="1" \
                    --saving_steps="5000" \
                    --per_device_train_batch_size="32" \
                    --per_device_eval_batch_size="32" \
                    --adam_beta1="0.9" \
                    --adam_beta2="0.98" \
                    --adam_epsilon="1e-06" \
                    --gradient_checkpointing \
                    --mask_time_prob="0.65" \
                    --mask_time_length="10" \
                    --push_to_hub 


TORCHDYNAMO_VERBOSE=1 accelerate launch sign2vec/train/run_sign2vec_pretraining.py \
                    --dataset_name="YoutubeASL" \
                    --run_name=$1-1 \
                    --hub_model_id="sign2vec-base-1" \
                    --tags sign2vec base yasl-norm single_cue \
                    --datasets "train" "test" \
                    --dataset_path="/ssd2/karahan/YASL/pose"  \
                    --model_config_file="experimental/configs/sign2vec_pretraining_config.yaml" \
                    --output_dir="/ssd2/karahan/YASL/sign2vec-base-1" \
                    --max_train_steps="20000" \
                    --num_warmup_steps="2000" \
                    --gradient_accumulation_steps="4" \
                    --learning_rate="0.0005" \
                    --weight_decay="0.001" \
                    --max_duration_in_seconds="20.0" \
                    --min_duration_in_seconds="2.0" \
                    --logging_steps="1" \
                    --saving_steps="5000" \
                    --per_device_train_batch_size="64" \
                    --per_device_eval_batch_size="64" \
                    --adam_beta1="0.9" \
                    --adam_beta2="0.98" \
                    --adam_epsilon="1e-06" \
                    --gradient_checkpointing \
                    --mask_time_prob="0.65" \
                    --mask_time_length="10" \
                    --push_to_hub 


TORCHDYNAMO_VERBOSE=1 accelerate launch sign2vec/train/run_sign2vec_pretraining.py \
                    --dataset_name="YoutubeASL" \
                    --run_name=$1-2 \
                    --hub_model_id="sign2vec-base-2" \
                    --tags sign2vec base yasl-norm single_cue \
                    --datasets "train" "test" \
                    --dataset_path="/ssd2/karahan/YASL/pose"  \
                    --model_config_file="experimental/configs/sign2vec_pretraining_config.yaml" \
                    --output_dir="/ssd2/karahan/YASL/sign2vec-base-2" \
                    --max_train_steps="20000" \
                    --num_warmup_steps="2000" \
                    --gradient_accumulation_steps="4" \
                    --learning_rate="0.0005" \
                    --weight_decay="0.001" \
                    --max_duration_in_seconds="20.0" \
                    --min_duration_in_seconds="2.0" \
                    --logging_steps="1" \
                    --saving_steps="5000" \
                    --max_gumbel_temperature="5.0" \
                    --per_device_train_batch_size="32" \
                    --per_device_eval_batch_size="32" \
                    --adam_beta1="0.9" \
                    --adam_beta2="0.98" \
                    --adam_epsilon="1e-06" \
                    --gradient_checkpointing \
                    --mask_time_prob="0.65" \
                    --mask_time_length="10" \
                    --push_to_hub 