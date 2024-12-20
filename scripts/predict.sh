python sign2vec/eval/prediction.py \
  --annotation_file=/path/to/annotations.json \
  --metadata_file=/path/to/yasl/data.json \
  --dataset_type=yasl \
  --batch_size=2 \
  --max_sequence_length=250 \
  --max_token_length=128 \
  --model_name=yasl-test \
  --max_val_samples=10 \
  --is_normalized \
  --skip_frames \
  --model_dir=/path/to/model.safetensors
