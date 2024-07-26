export OUTPUT_DIR=/ssd1/karahan/

mkdir -p ${OUTPUT_DIR}/How2Sign/
mkdir -p ${OUTPUT_DIR}/How2Sign/tti

python3 sign2vec/utils/run_sign2vec_inference_tti.py --model_name karahansahin/sign2vec-yasl-mc-sc-64-2-d1 \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/tti \
                                                 --input_file H2S_train.h5 \
                                                 --output_file sign2vec.train.0.h5 \
                                                 --metadata_file metadata_sign2vec.train.json \
                                                 --annotation_file datasets/H2S.annotations.train.json

python3 sign2vec/utils/run_sign2vec_inference_tti.py --model_name karahansahin/sign2vec-yasl-mc-sc-64-2-d1 \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/tti \
                                                 --input_file H2S_val.h5 \
                                                 --output_file sign2vec.dev.0.h5 \
                                                 --metadata_file metadata_sign2vec.dev.json \
                                                 --annotation_file datasets/H2S.annotations.dev.json

python3 sign2vec/utils/run_sign2vec_inference_tti.py --model_name karahansahin/sign2vec-yasl-mc-sc-64-2-d1 \
                                                 --data_dir ${OUTPUT_DIR}/How2Sign \
                                                 --output_path ${OUTPUT_DIR}/How2Sign/tti \
                                                 --input_file H2S_test.h5 \
                                                 --output_file sign2vec.test.0.h5 \
                                                 --metadata_file metadata_sign2vec.test.json \
                                                 --annotation_file datasets/H2S.annotations.test.json




tar -czvf /ssd1/karahan/h2_t5_tti.tar.gz /ssd1/karahan/tti
