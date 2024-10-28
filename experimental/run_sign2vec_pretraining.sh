# get bash parameter run_name
run_name=$1
cuda_device=$2
export CUDA_VISIBLE_DEVICES=$cuda_device

TORCHDYNAMO_VERBOSE=1 accelerate launch sign2vec/train/run_sign2vec_pretraining.py \
                    --dataset_name="YoutubeASL" \
                    --run_name="sign2vec-base-linear" \
                    --hub_model_id="sign2vec-base-linear" \
                    --tags sign2vec base yasl-norm single_cue \
                    --datasets "train" "test" \
                    --dataset_path="/ssd2/karahan/YASL/pose"  \
                    --model_config_file="experimental/configs/s2v_multi_code.yaml" \
                    --output_dir="/ssd2/karahan/YASL/sign2vec-base-linear" \
                    --max_train_steps="10000" \
                    --num_warmup_steps="1000" \
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
                    --run_name="sign2vec-base-linear-zero-mean" \
                    --hub_model_id="sign2vec-base-linear-zero-mean" \
                    --tags sign2vec base yasl-norm single_cue \
                    --datasets "train" "test" \
                    --dataset_path="/ssd2/karahan/YASL/pose"  \
                    --model_config_file="experimental/configs/s2v_multi_code.yaml" \
                    --output_dir="/ssd2/karahan/YASL/sign2vec-base-linear-zero-mean" \
                    --max_train_steps="10000" \
                    --num_warmup_steps="1000" \
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
                    --zero_mean \
                    --push_to_hub 


TORCHDYNAMO_VERBOSE=1 accelerate launch sign2vec/train/run_sign2vec_pretraining.py \
                    --dataset_name="YoutubeASL" \
                    --run_name="sign2vec-base-linear-1-512" \
                    --hub_model_id="sign2vec-base-linear-1-512" \
                    --tags sign2vec base yasl-norm single_cue \
                    --datasets "train" "test" \
                    --dataset_path="/ssd2/karahan/YASL/pose"  \
                    --model_config_file="experimental/configs/s2v_single_code.yaml" \
                    --output_dir="/ssd2/karahan/YASL/sign2vec-base-linear" \
                    --max_train_steps="10000" \
                    --num_warmup_steps="1000" \
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
                    --run_name="sign2vec-base-linear-zero-mean-1-512" \
                    --hub_model_id="sign2vec-base-linear-zero-mean-1-512" \
                    --tags sign2vec base yasl-norm single_cue \
                    --datasets "train" "test" \
                    --dataset_path="/ssd2/karahan/YASL/pose"  \
                    --model_config_file="experimental/configs/s2v_single_code.yaml" \
                    --output_dir="/ssd2/karahan/YASL/sign2vec-base-linear-zero-mean-1-512" \
                    --max_train_steps="10000" \
                    --num_warmup_steps="1000" \
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
                    --zero_mean \
                    --push_to_hub 
