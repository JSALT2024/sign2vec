python prediction.py \
    --model_dir data/ \
    --model_name yasl-test \
    --dataset_type yasl \
    --dataset_dir /data/YoutubeASL/subset \
    --annotation_file /data/YoutubeASL/features \
    --metadata_file /data/YoutubeASL/features_v2/keypoints \
    --output_file /results/predictions.json \
    --verbose

python predict.py --model_dir ./results --model_name h2s-test --dataset_type how2sign --dataset_dir /path/to/how2sign/data --annotation_file /path/to/annotations.json --metadata_file /path/to/metadata.json --output_file ./results/h2s-test/predictions.json --verbose