TORCHDYNAMO_VERBOSE=1 accelerate launch pretraining/run_sign2vec_pretraining.py \
                                        --model_name_or_path="patrickvonplaten/wav2vec2-base-v2" \
                                        --output_dir="./sign2vec" \
                                        --env="server" \
                                        --max_train_steps="20000" \
                                        --num_warmup_steps="3000" \
                                        --gradient_accumulation_steps="4" \
                                        --learning_rate="0.001" \
                                        --weight_decay="0.01" \
                                        --max_duration_in_seconds="20.0" \
                                        --min_duration_in_seconds="2.0" \
                                        --logging_steps="1" \
                                        --saving_steps="10000" \
                                        --per_device_train_batch_size="32" \
                                        --per_device_eval_batch_size="32" \
                                        --adam_beta1="0.9" \
                                        --adam_beta2="0.98" \
                                        --adam_epsilon="1e-06" \
                                        --gradient_checkpointing \
                                        --mask_time_prob="0.65" \
                                        --mask_time_length="10" \
                                        --use_face \
                                        --use_hands \
                                        --use_pose \
                                        --train_data_path="/ssd2/karahan/YASL/train_dataset.csv" \
                                        --validation_data_path="/ssd2/karahan/YASL/val_dataset.csv" \
                                        --data_dir="/ssd2/karahan/YASL" \
                                        --config_name="pretraining/config.json" \
                                        --push_to_hub \
                                        --hub_model_id="sign2vec-v0-yasl"