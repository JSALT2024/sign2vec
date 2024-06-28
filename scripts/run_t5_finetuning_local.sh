python3 finetune/run_finetuning.py  --t5_model_path_or_name='google-t5/t5-small' \
                                    --sign2vec_model_path='karahansahin/sign2vec-v0-how2sign' \
                                    --data_path='pretraining/how2sign/' \
                                    --output_dir='output' \
                                    --batch_size="8" \
                                    --sign2vec_embed_dim="768" \
                                    --t5_embed_dim="512" \
                                    --dropout="0.1" \
                                    --mask_time_prob="0.65" \
                                    --mask_time_length="10" \
                                    --max_frames="500" \
                                    --max_epochs="100" \
                                    --learning_rate="1e-4" \
                                    --warmup_steps="1000" \
                                    --weight_decay="0.01" \
                                    --wandb_project_name='sign2vec_how2sign' \
                                    --wandb_run_name='sign2vec_how2sign_v0.0' \
                                    --train_dataset_path='pretraining/how2sign/training.csv' \
                                    --val_dataset_path='pretraining/how2sign/training.csv' \
                                    --test_dataset_path='pretraining/how2sign/training.csv' \
                                    --token='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'  