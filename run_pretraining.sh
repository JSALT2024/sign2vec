

TORCHDYNAMO_VERBOSE=1 accelerate launch pretraining/run_sign2vec_pretraining.py \
                                                --model_name_or_path="patrickvonplaten/wav2vec2-base-v2" \
                                                --output_dir="./sign2vec" \
                                                --max_train_steps="20000" \
                                                --num_warmup_steps="32000" \
                                                --gradient_accumulation_steps="8" \
                                                --learning_rate="0.001" \
                                                --weight_decay="0.01" \
                                                --max_duration_in_seconds="20.0" \
                                                --min_duration_in_seconds="2.0" \
                                                --logging_steps="1" \
                                                --saving_steps="10000" \
                                                --per_device_train_batch_size="16" \
                                                --per_device_eval_batch_size="16" \
                                                --adam_beta1="0.9" \
                                                --adam_beta2="0.98" \
                                                --adam_epsilon="1e-06" \
                                                --gradient_checkpointing \
                                                --mask_time_prob="0.65" \
                                                --mask_time_length="10" \
                                                --use_face \
                                                --use_hands \
                                                --use_pose \
                                                --train_info_path="sign2vec/config/info.json" \
                                                --train_data_path="sign2vec/features" \
                                                --validation_info_path="sign2vec/config/info.json" \
                                                --validation_data_path="sign2vec/features" \
                                                --config_name="pretraining/config.json"
                                                --push_to_hub \
                                                --hub_model_id="sign2vec" 

