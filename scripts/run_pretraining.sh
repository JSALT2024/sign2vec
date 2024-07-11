TORCHDYNAMO_VERBOSE=1 accelerate launch pretraining/run_sign2vec_pretraining.py \
                                        --model_name_or_path="patrickvonplaten/wav2vec2-base-v2" \
                                        --output_dir="./sign2vec" \
                                        --env="server" \
                                         --dataset_name="how2sign" \
                                        --sampling_rate="25" \
                                        --output_dir="./sign2vec" \
                                        --num_train_epochs="1000" \
                                        --max_train_steps="200000" \
                                        --num_warmup_steps="30000" \
                                        --gradient_accumulation_steps="4" \
                                        --learning_rate="0.001" \
                                        --weight_decay="0.01" \
                                        --max_duration_in_seconds="10.0" \
                                        --min_duration_in_seconds="2.0" \
                                        --logging_steps="1" \
                                        --saving_steps="10000" \
                                        --per_device_train_batch_size="16" \
                                        --per_device_eval_batch_size="16" \
                                        --adam_beta1="0.9" \
                                        --adam_beta2="0.98" \
                                        --adam_epsilon="1e-06" \
                                        --gradient_checkpointing \
                                        --mask_time_prob="0.5" \
                                        --mask_time_length="10" \
                                        --train_data_path="how2sign/training.csv" \
                                        --validation_data_path="how2sign/validation.csv" \
                                        --data_dir="../../../ssd1/karahan/" \
                                        --config_name="pretraining/config.json" \
                                        --push_to_hub \
                                        --hub_model_id="sign2vec-v0-how2sign"