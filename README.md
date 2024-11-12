# sign2vec

The implementation of `wav2vec2.0` for SLR

* copy how2sign from Royal

```
scp -J karahan@193.140.195.142 karahan@193.140.195.17:/ssd1/karahan/How2Sign/H2S_train.h5 /home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/How2Sign/Mediapipe/

scp -J karahan@193.140.195.142 karahan@193.140.195.17:/ssd1/karahan/How2Sign/H2S_val.h5 /home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/How2Sign/Mediapipe/

scp -J karahan@193.140.195.142 karahan@193.140.195.17:/ssd1/karahan/How2Sign/H2S_test.h5 /home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/How2Sign/Mediapipe/
```

## Usage


### 1. Sign2Vec Pretraining

```bash
accelerate launch sign2vec/train/run_sign2vec_pretraining.py \
                    --dataset_name="YoutubeASL" \
                    --tags sign2vec base v0.0 single_cue dev \
                    --datasets "train" "test" \
                    --dataset_path="/ssd2/karahan/YASL/pose"  \
                    --model_config_file="experimental/configs/sign2vec_pretraining_config.yaml" \
                    --output_dir="./sign2vec-base-v0.0" \
                    --max_train_steps="20000" \
                    --num_warmup_steps="32000" \
                    --gradient_accumulation_steps="8" \
                    --learning_rate="0.005" \
                    --weight_decay="0.01" \
                    --max_duration_in_seconds="20.0" \
                    --min_duration_in_seconds="2.0" \
                    --logging_steps="1" \
                    --saving_steps="10000" \
                    --per_device_train_batch_size="8" \
                    --per_device_eval_batch_size="8" \
                    --adam_beta1="0.9" \
                    --adam_beta2="0.98" \
                    --adam_epsilon="1e-06" \
                    --gradient_checkpointing \
                    --mask_time_prob="0.65" \
                    --mask_time_length="10" \
                    --push_to_hub 
```

### 2. T5 Training

In this training, you can either utilize raw pose to train your modals or you can use pretrained ``sign2vec`` model for training. 

The parameters in the pre-training and fine-tuning are coming from original YoutubeASL paper to replicate the experimental setup.

In either case, you are file structure should look like this.

```markdown
YASL
|---yasl.train.csv
|---yasl.dev.csv
|---yasl.test.csv
|---mae
|     |--- yasl_mae_0.h5
|     |--- ....
|---sign2vec
|     |--- yasl_sign2vec_0.h5
|     |--- ....
|---pose
|     |--- yasl_pose_0.h5
|     |--- ....
```

#### 2a. YASL T5 Pretraining

```bash
python3 -m sign2vec.train.run_finetuning --annotation_file='/home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/YASL' \
                                         --metadata_file='/home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/YASL/keypoints/' \
                                         --dataset_type='yasl' \
                                         --model_id="google-t5/t5-base" \
                                         --max_training_step="20000" \
                                         --per_device_train_batch_size="32" \
                                         --per_device_eval_batch_size=2
                                         --gradient_accumulation_steps=4 \
                                         --eval_steps=2 \
                                         --learning_rate=0.001 \
                                         --max_sequence_length=256 \
                                         --max_token_length=128 \
                                         --model_name=h2s-pose-no-norm
                                         --skip_frames \
                                         --logging_steps=1 \
                                         --eval_steps="1000"
```

#### 2b. How2Sign Finetuning

```bash
python3 -m sign2vec.train.run_finetuning --dataset_dir=/ssd1/karahan/How2Sign \
                                          --modality="pose" \
                                          --model_id="google-t5/t5-base" \
                                          --max_training_step="20000" \
                                          --learning_rate="0.001" \
                                          --max_sequence_length="256" \
                                          --max_token_length="128" \
                                          --per_device_train_batch_size="32" \
                                          --gradient_accumulation_steps="4" \
                                          --skip_frames \
                                          --per_device_eval_batch_size="2" \
                                          --logging_steps="10" \
                                          --eval_steps="1000" \
                                          --model_name=h2s-pose-no-norm
```

