# get bash parameter run_name
run_name=$1
cuda_device=$2
export CUDA_VISIBLE_DEVICES=$cuda_device


TORCHDYNAMO_VERBOSE=1 accelerate launch sign2vec/train/run_sign2vec_pretraining.py \
                    --dataset_name="YoutubeASL" \
                    --run_name="sign2vec-base-mixed-mcue-zero-mean-div-1.0" \
                    --hub_model_id="sign2vec-base-mixed-mcue-zero-mean-div-1.0" \
                    --tags sign2vec base yasl-norm single_cue \
                    --datasets "train" "dev" \
                    --dataset_path="/ssd2/karahan/YASL/pose"  \
                    --model_config_file="experimental/configs/s2v_multi_code_mixed_mcue.yaml" \
                    --metadata_file='/ssd2/karahan/YASL/pose' \
                    --annotation_file='/ssd2/karahan/YASL/' \
                    --output_dir="/ssd2/karahan/YASL/sign2vec-base-mixed-mcue-zero-mean-div-1.0" \
                    --max_train_steps="10000" \
                    --num_warmup_steps="1000" \
                    --gradient_accumulation_steps="8" \
                    --learning_rate="0.0005" \
                    --weight_decay="0.001" \
                    --max_duration_in_seconds="20.0" \
                    --min_duration_in_seconds="2.0" \
                    --logging_steps="1" \
                    --saving_steps="5000" \
                    --per_device_train_batch_size="16" \
                    --per_device_eval_batch_size="16" \
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
                    --run_name="sign2vec-base-linear-mcue-zero-mean-div-1.0" \
                    --hub_model_id="sign2vec-base-linear-mcue-zero-mean-div-1.0" \
                    --tags sign2vec base yasl-norm single_cue \
                    --datasets "train" "dev" \
                    --dataset_path="/ssd2/karahan/YASL/pose"  \
                    --model_config_file="experimental/configs/s2v_multi_code_linear_mcue.yaml" \
                    --metadata_file='/ssd2/karahan/YASL/pose' \
                    --annotation_file='/ssd2/karahan/YASL/' \
                    --output_dir="/ssd2/karahan/YASL/sign2vec-base-linear-mcue-zero-mean-div-1.0" \
                    --max_train_steps="10000" \
                    --num_warmup_steps="1000" \
                    --gradient_accumulation_steps="8" \
                    --learning_rate="0.0005" \
                    --weight_decay="0.001" \
                    --max_duration_in_seconds="20.0" \
                    --min_duration_in_seconds="2.0" \
                    --logging_steps="1" \
                    --saving_steps="5000" \
                    --per_device_train_batch_size="16" \
                    --per_device_eval_batch_size="16" \
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
                    --run_name="sign2vec-base-conv-mcue-zero-mean-div-1.0" \
                    --hub_model_id="sign2vec-base-conv-mcue-zero-mean-div-1.0" \
                    --tags sign2vec base yasl-norm single_cue \
                    --datasets "train" "dev" \
                    --dataset_path="/ssd2/karahan/YASL/pose"  \
                    --model_config_file="experimental/configs/s2v_multi_code_conv_mcue.yaml" \
                    --metadata_file='/ssd2/karahan/YASL/pose' \
                    --annotation_file='/ssd2/karahan/YASL/' \
                    --output_dir="/ssd2/karahan/YASL/sign2vec-base-conv-mcue-zero-mean-div-1.0" \
                    --max_train_steps="10000" \
                    --num_warmup_steps="1000" \
                    --gradient_accumulation_steps="8" \
                    --learning_rate="0.0005" \
                    --weight_decay="0.001" \
                    --max_duration_in_seconds="20.0" \
                    --min_duration_in_seconds="2.0" \
                    --logging_steps="1" \
                    --saving_steps="5000" \
                    --per_device_train_batch_size="16" \
                    --per_device_eval_batch_size="16" \
                    --adam_beta1="0.9" \
                    --adam_beta2="0.98" \
                    --adam_epsilon="1e-06" \
                    --gradient_checkpointing \
                    --mask_time_prob="0.65" \
                    --mask_time_length="10" \
                    --zero_mean \
                    --push_to_hub 


# TORCHDYNAMO_VERBOSE=1 accelerate launch sign2vec/train/run_sign2vec_pretraining.py \
#                     --dataset_name="YoutubeASL" \
#                     --run_name="sign2vec-base-mixed-mcue-zero-mean-div-1.0-c128" \
#                     --hub_model_id="sign2vec-base-mixed-mcue-zero-mean-div-1.0-c128" \
#                     --tags sign2vec base yasl-norm single_cue \
#                     --datasets "train" "dev" \
#                     --dataset_path="/ssd2/karahan/YASL/pose"  \
#                     --model_config_file="experimental/configs/s2v_multi_code_mixed_n_mcue.yaml" \
#                     --metadata_file='/ssd2/karahan/YASL/pose' \
#                     --annotation_file='/ssd2/karahan/YASL/' \
#                     --output_dir="/ssd2/karahan/YASL/sign2vec-base-mixed-mcue-zero-mean-div-1.0-c128" \
#                     --max_train_steps="10000" \
#                     --num_warmup_steps="1000" \
#                     --gradient_accumulation_steps="8" \
#                     --learning_rate="0.0005" \
#                     --weight_decay="0.001" \
#                     --max_duration_in_seconds="20.0" \
#                     --min_duration_in_seconds="2.0" \
#                     --logging_steps="1" \
#                     --saving_steps="5000" \
#                     --per_device_train_batch_size="16" \
#                     --per_device_eval_batch_size="16" \
#                     --adam_beta1="0.9" \
#                     --adam_beta2="0.98" \
#                     --adam_epsilon="1e-06" \
#                     --gradient_checkpointing \
#                     --mask_time_prob="0.65" \
#                     --mask_time_length="10" \
#                     --zero_mean \
#                     --push_to_hub 

# TORCHDYNAMO_VERBOSE=1 accelerate launch sign2vec/train/run_sign2vec_pretraining.py \
#                     --dataset_name="YoutubeASL" \
#                     --run_name="sign2vec-base-linear-mcue-zero-mean-div-1.0-c128" \
#                     --hub_model_id="sign2vec-base-linear-mcue-zero-mean-div-1.0-c128" \
#                     --tags sign2vec base yasl-norm single_cue \
#                     --datasets "train" "dev" \
#                     --dataset_path="/ssd2/karahan/YASL/pose"  \
#                     --model_config_file="experimental/configs/s2v_multi_code_linear_n_mcue.yaml" \
#                     --metadata_file='/ssd2/karahan/YASL/pose' \
#                     --annotation_file='/ssd2/karahan/YASL/' \
#                     --output_dir="/ssd2/karahan/YASL/sign2vec-base-linear-mcue-zero-mean-div-1.0-c128" \
#                     --max_train_steps="10000" \
#                     --num_warmup_steps="1000" \
#                     --gradient_accumulation_steps="8" \
#                     --learning_rate="0.0005" \
#                     --weight_decay="0.001" \
#                     --max_duration_in_seconds="20.0" \
#                     --min_duration_in_seconds="2.0" \
#                     --logging_steps="1" \
#                     --saving_steps="5000" \
#                     --per_device_train_batch_size="16" \
#                     --per_device_eval_batch_size="16" \
#                     --adam_beta1="0.9" \
#                     --adam_beta2="0.98" \
#                     --adam_epsilon="1e-06" \
#                     --gradient_checkpointing \
#                     --mask_time_prob="0.65" \
#                     --mask_time_length="10" \
#                     --zero_mean \
#                     --push_to_hub 

# TORCHDYNAMO_VERBOSE=1 accelerate launch sign2vec/train/run_sign2vec_pretraining.py \
#                     --dataset_name="YoutubeASL" \
#                     --run_name="sign2vec-base-conv-mcue-zero-mean-div-1.0-c128" \
#                     --hub_model_id="sign2vec-base-conv-mcue-zero-mean-div-1.0-c128" \
#                     --tags sign2vec base yasl-norm single_cue \
#                     --datasets "train" "dev" \
#                     --dataset_path="/ssd2/karahan/YASL/pose"  \
#                     --model_config_file="experimental/configs/s2v_multi_code_conv_n_mcue.yaml" \
#                     --metadata_file='/ssd2/karahan/YASL/pose' \
#                     --annotation_file='/ssd2/karahan/YASL/' \
#                     --output_dir="/ssd2/karahan/YASL/sign2vec-base-conv-mcue-zero-mean-div-1.0-c128" \
#                     --max_train_steps="10000" \
#                     --num_warmup_steps="1000" \
#                     --gradient_accumulation_steps="8" \
#                     --learning_rate="0.0005" \
#                     --weight_decay="0.001" \
#                     --max_duration_in_seconds="20.0" \
#                     --min_duration_in_seconds="2.0" \
#                     --logging_steps="1" \
#                     --saving_steps="5000" \
#                     --per_device_train_batch_size="16" \
#                     --per_device_eval_batch_size="16" \
#                     --adam_beta1="0.9" \
#                     --adam_beta2="0.98" \
#                     --adam_epsilon="1e-06" \
#                     --gradient_checkpointing \
#                     --mask_time_prob="0.65" \
#                     --mask_time_length="10" \
#                     --zero_mean \
#                     --push_to_hub 