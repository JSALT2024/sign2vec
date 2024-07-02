NCCL_IB_DISABLE="1" NCCL_P2P_DISABLE="1" python3 finetune/run_hf_finetuning.py  --t5_model_path_or_name='goog>
                                                --train_data_path="how2sign/training.csv" \
                                                --val_data_path="how2sign/test.csv" \
                                                --test_data_path="how2sign/test.csv" \
                                                --output_dir="./outputs" \
                                                --data_dir="../../../../ssd1/karahan/" \
                                                --per_device_train_batch_size="8" \
                                                --per_device_eval_batch_size="8" \
                                                --learning_rate="5e-5" \
                                                --num_train_epochs="100" \
                                                --weight_decay="0.01" \
                                                --evaluation_strategy="epoch" \
                                                --save_total_limit="100" \
                                                --logging_dir="./logs" \
                                                --debug \



