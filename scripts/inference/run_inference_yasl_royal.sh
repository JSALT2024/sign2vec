rm -rf /ssd2/karahan/YASL/sign2vec.train.0.h5

# python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/sign2vec-yasl-sc-sc-64-8-d1 \
#                                                  --data_dir /ssd2/karahan/YASL \
#                                                  --output_path /ssd2/karahan/YASL \
#                                                  --input_file YouTubeASL.keypoints.train.0.h5 \
#                                                  --output_file sign2vec.train.0.h5 \
#                                                  --metadata_file metadata_sign2vec.train.json \
#                                                  --annotation_file /ssd2/karahan/YASL/yasl.annotations.train.json \
#                                                  --use_shards \
#                                                  --shard_prefix 'yasl_pose'


python3 sign2vec/utils/run_sign2vec_inference.py --model_name karahansahin/sign2vec-yasl-sc-sc-64-8-d1 \
                                                 --data_dir /ssd2/karahan/YASL/sign2vec/ \
                                                 --output_path /ssd2/karahan/YASL/sign2vec/ \
                                                 --input_file YouTubeASL.keypoints.train.0.h5 \
                                                 --output_file sign2vec.dev.0.h5 \
                                                 --metadata_file metadata_sign2vec.dev.json \
                                                 --annotation_file /ssd2/karahan/YASL/YT.annotations.dev.json \
                                                 --use_shards \
                                                 --shard_prefix 'yasl_pose'

