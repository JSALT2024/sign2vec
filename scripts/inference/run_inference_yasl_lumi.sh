mkdir /pfs/lustrep2/scratch/project_465000977/data/YoutubeASL/features/sign2vec

python3 sign2vec/utils/run_sign2vec_inference_yasl.py --model_name karahansahin/sign2vec-yasl-sc-sc-64-8-d1 \
                                                 --data_dir pfs/lustrep2/scratch/project_465000977/data/YoutubeASL/features/keypoints \
                                                 --output_path pfs/lustrep2/scratch/project_465000977/data/YoutubeASL/features/sign2vec \
                                                 --input_file YouTubeASL.keypoints.train.0.h5 \
                                                 --output_file sign2vec.train.0.h5 \
                                                 --metadata_file metadata_sign2vec.train.json \
                                                 --annotation_file path/to/annotation/file


