export OUTPUT_DIR=/ssd1/karahan/

rm -rf ${OUTPUT_DIR}/How2Sign/pose/pose.train.0.h5
rm -rf ${OUTPUT_DIR}/How2Sign/pose/pose.dev.0.h5
rm -rf ${OUTPUT_DIR}/How2Sign/pose/pose.test.0.h5

# mkdir -p ${OUTPUT_DIR}/How2Sign/

python3 sign2vec/utils/run_pose_inference.py --model_name karahansahin/sign2vec-yasl-mc-sc-64-2-d1 \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/pose \
                                                 --input_file H2S_train.h5 \
                                                 --output_file pose.train.0.h5 \
                                                 --metadata_file metadata_pose.train.json \
                                                 --annotation_file datasets/H2S.annotations.train.json

python3 sign2vec/utils/run_pose_inference.py --model_name karahansahin/sign2vec-yasl-mc-sc-64-2-d1 \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/pose \
                                                 --input_file H2S_val.h5 \
                                                 --output_file pose.dev.0.h5 \
                                                 --metadata_file metadata_pose.dev.json \
                                                 --annotation_file datasets/H2S.annotations.dev.json

python3 sign2vec/utils/run_pose_inference.py --model_name karahansahin/sign2vec-yasl-mc-sc-64-2-d1 \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/pose \
                                                 --input_file H2S_test.h5 \
                                                 --output_file pose.test.0.h5 \
                                                 --metadata_file metadata_pose.test.json \
                                                 --annotation_file datasets/H2S.annotations.test.json

